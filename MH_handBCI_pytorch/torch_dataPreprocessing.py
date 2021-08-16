import numpy as np
import random
import scipy.io
import sys
import os
import glob
import torch
from torch.utils.data import Dataset
import torch.nn.functional as F
from tfrecord.torch.dataset import MultiTFRecordDataset
from scipy.ndimage.filters import gaussian_filter1d

def loadAllRealDatasets(args):
        """
        Loads the labels & data for each day specified in the training args, and returns the relevant variables as data cubes.
        Also collects the file names of all .tfrecord files needed for including the synthetic data.
        """
        neuralCube_all = []
        targets_all = []
        errWeights_all = []
        numBinsPerTrial_all = []
        cvIdx_all = []
        recordFileSet_all = []
    
        for dayIdx in range(args['nDays']):
            neuralData, targets, errWeights, binsPerTrial, cvIdx = prepareDataCubesForRNN(args['sentencesFile_'+str(dayIdx)],
                                                                                          args['singleLettersFile_'+str(dayIdx)],
                                                                                          args['labelsFile_'+str(dayIdx)],
                                                                                          args['cvPartitionFile_'+str(dayIdx)],
                                                                                          args['sessionName_'+str(dayIdx)],
                                                                                          args['rnnBinSize'],
                                                                                          args['timeSteps'],
                                                                                          args['isTraining'])

            neuralCube_all.append(neuralData)
            targets_all.append(targets)
            errWeights_all.append(errWeights)
            numBinsPerTrial_all.append(binsPerTrial)
            cvIdx_all.append(cvIdx)

            synthDir = args['syntheticDatasetDir_'+str(dayIdx)]
            if os.path.isdir(synthDir):
                recordFileSet = [os.path.join(synthDir, file) for file in os.listdir(synthDir)]
            else:
                recordFileSet = []
            
            if args['synthBatchSize']>0 and len(recordFileSet)==0:
                sys.exit('Error! No synthetic files found in directory ' + args['syntheticDatasetDir_'+str(dayIdx)] + ', exiting.')
                         
            random.shuffle(recordFileSet)
            recordFileSet_all.append(recordFileSet)
                                 
        return neuralCube_all, targets_all, errWeights_all, numBinsPerTrial_all, cvIdx_all, recordFileSet_all    
  
def normalizeSentenceDataCube(sentenceDat, singleLetterDat):
    """
    Normalizes the neural data cube by subtracting means and dividing by the standard deviation. 
    Important: we use means and standard deviations from the single letter data. This is needed since we 
    initialize the HMM parameters using the single letter data, so the sentence data needs to be normalized in the same way. 
    """
    neuralCube = sentenceDat['neuralActivityCube'].astype(np.float64)

    #subtract block-specific means from each trial to counteract the slow drift in feature means over time
    for b in range(sentenceDat['blockList'].shape[0]):
        trialsFromThisBlock = np.squeeze(sentenceDat['sentenceBlockNums']==sentenceDat['blockList'][b])
        trialsFromThisBlock = np.argwhere(trialsFromThisBlock)

        closestIdx = np.argmin(np.abs(singleLetterDat['blockList'].astype(np.int32) - sentenceDat['blockList'][b].astype(np.int32)))
        blockMeans = singleLetterDat['meansPerBlock'][closestIdx,:]

        neuralCube[trialsFromThisBlock,:,:] -= blockMeans[np.newaxis,np.newaxis,:]

    #divide by standard deviation to normalize the units
    neuralCube = neuralCube / singleLetterDat['stdAcrossAllData'][np.newaxis,:,:]
    
    return neuralCube

def prepareDataCubesForRNN(sentenceFile, singleLetterFile, labelFile, cvPartitionFile, sessionName, rnnBinSize, nTimeSteps, isTraining):
    """
    Loads raw data & HMM labels and returns training and validation data cubes for RNN training (or inference). 
    Normalizes the neural activity using the single letter means & standard deviations.
    Does some additional pre-processing, including zero-padding the data and cutting off the end of the last character if it is too long.
    (Long pauses occur at the end of some sentences since T5 often paused briefly after finishing instead of 
    continuing immediately to the next sentence).
    """
    sentenceDat = scipy.io.loadmat(sentenceFile)
    slDat = scipy.io.loadmat(singleLetterFile)
    labelsDat = scipy.io.loadmat(labelFile)
    cvPart = scipy.io.loadmat(cvPartitionFile)
                      
    errWeights = 1-labelsDat['ignoreErrorHere']
    charProbTarget = labelsDat['charProbTarget']
    charStartTarget = labelsDat['charStartTarget'][:,:,np.newaxis]

    #Here we update the error weights to ignore time bins outside of the sentence
    for t in range(labelsDat['timeBinsPerSentence'].shape[0]):
        errWeights[t,labelsDat['timeBinsPerSentence'][t,0]:] = 0

        #Also, we cut off the end of the trial if there is a very long pause after the last letter - this could hurt
        #training. 
        maxPause = 150
        lastCharStart = np.argwhere(charStartTarget[t,:]>0.5)
        errWeights[t,(lastCharStart[-1,0]+maxPause):] = 0
        labelsDat['timeBinsPerSentence'][t,0] = (lastCharStart[-1,0]+maxPause)

    #For convenience, we combine the two targets.
    #The rest of the code then assumes that the last column is the character start target.
    combinedTargets = np.concatenate([charProbTarget, charStartTarget], axis=2)

    nRNNOutputs = combinedTargets.shape[2] 
    binsPerTrial = np.round(labelsDat['timeBinsPerSentence']/rnnBinSize).astype(np.int32)
    binsPerTrial = np.squeeze(binsPerTrial)

    #get normalized neural data cube for the sentences
    neuralData = normalizeSentenceDataCube(sentenceDat, slDat)

    #bin the data across the time axis
    if rnnBinSize>1:
        neuralData = binTensor(neuralData, rnnBinSize)
        combinedTargets = binTensor(combinedTargets, rnnBinSize)
        errWeights = np.squeeze(binTensor(errWeights[:,:,np.newaxis], rnnBinSize))

    #zero padding
    if isTraining:
        #train mode, add some extra zeros to the end so that we can begin snippets near the end of sentences
        edgeSpace = (nTimeSteps-100)
        padTo = neuralData.shape[1]+edgeSpace*2
        
        padNeuralData = np.zeros([neuralData.shape[0], padTo, neuralData.shape[2]])
        padCombinedTargets = np.zeros([combinedTargets.shape[0], padTo, combinedTargets.shape[2]])
        padErrWeights = np.zeros([errWeights.shape[0], padTo])

        padNeuralData[:,edgeSpace:(edgeSpace+neuralData.shape[1]),:] = neuralData
        padCombinedTargets[:,edgeSpace:(edgeSpace+combinedTargets.shape[1]),:] = combinedTargets
        padErrWeights[:,edgeSpace:(edgeSpace+errWeights.shape[1])] = errWeights
    else:
        #inference mode, pad up to the specified time steps (which should be > than the data cube length, and a multiple of skipLen)
        padTo = nTimeSteps

        padNeuralData = np.zeros([neuralData.shape[0], padTo, neuralData.shape[2]])
        padCombinedTargets = np.zeros([combinedTargets.shape[0], padTo, combinedTargets.shape[2]])
        padErrWeights = np.zeros([errWeights.shape[0], padTo])

        padNeuralData[:,0:neuralData.shape[1],:] = neuralData
        padCombinedTargets[:,0:combinedTargets.shape[1],:] = combinedTargets
        padErrWeights[:,0:errWeights.shape[1]] = errWeights

    #gather the train/validation fold indices
    cvIdx = {}                          
    cvIdx['trainIdx'] = np.squeeze(cvPart[sessionName+'_train'])
    cvIdx['testIdx'] = np.squeeze(cvPart[sessionName+'_test'])

    return padNeuralData, padCombinedTargets, padErrWeights, binsPerTrial, cvIdx

def binTensor(data, binSize):
    """
    A simple utility function to bin a 3d numpy tensor along axis 1 (the time axis here). Data is binned by
    taking the mean across a window of time steps. 
    
    Args:
        data (tensor : B x T x N): A 3d tensor with batch size B, time steps T, and number of features N
        binSize (int): The bin size in # of time steps
        
    Returns:
        binnedTensor (tensor : B x S x N): A 3d tensor with batch size B, time bins S, and number of features N.
                                           S = floor(T/binSize)
    """
    
    nBins = np.floor(data.shape[1]/binSize).astype(int)
    
    sh = np.array(data.shape)
    sh[1] = nBins
    binnedTensor = np.zeros(sh)
    
    binIdx = np.arange(0,binSize).astype(int)
    for t in range(nBins):
        binnedTensor[:,t,:] = np.mean(data[:,binIdx,:],axis=1)
        binIdx += binSize
    
    return binnedTensor

def gaussSmooth(inputs,kernelSD):
    """
    Applies a 1D gaussian smoothing operation with tensorflow to smooth the data along the time axis.
    
    Args:
        inputs (tensor : B x T x N): A 3d tensor with batch size B, time steps T, and number of features N
        kernelSD (float): standard deviation of the Gaussian smoothing kernel
        
    Returns:
        smoothedData (tensor : B x T x N): A smoothed 3d tensor with batch size B, time steps T, and number of features N
    """
    ## use Pytorch's functional API F.conv1d                             
    #get gaussian smoothing kernel
    inp = np.zeros([100])
    inp[50] = 1
    gaussKernel = gaussian_filter1d(inp, kernelSD)

    validIdx = np.argwhere(gaussKernel>0.01)
    gaussKernel = gaussKernel[validIdx]
    gaussKernel = np.squeeze(gaussKernel/np.sum(gaussKernel))
    weights = torch.tensor(gaussKernel)
    weights = weights.view(1,1,len(weights)).repeat(inputs.shape[0], 1,1)
    #apply the convolution separately for each feature
    convOut = np.zeros_like(inputs)
    for x in range(inputs.shape[2]):
        data = inputs[:,:,x]
        data = torch.tensor(data[:,:,np.newaxis]).transpose(1,2)
        output = F.conv1d(data, weights, padding='same', groups = 1)[:,0,:]
        convOut[:,:,x] = output
    return convOut

class handBCI_SythDataset(object):
    """hanBCI sythetic dataset per day"""

    def __init__(self, tfrDir, args):
        """
        Args:
           tfrDir: tensorRecord file directory
           args: dictionary of all arguments
        """
#         tfrDir=rootDir+'/Datasets/t5.2019.12.09/HeldOutTrials/t5.2019.12.09_syntheticSentences/'
        self.tfrecord_pattern = tfrDir+"{}.tfrecord"
        self.index_pattern = None
        tfrfilesNames = [f.split('\\')[-1] for f in glob.glob(tfrDir+'*.tfrecord')]
        self.splits = {}
        for tr in tfrfilesNames:
            self.splits[tr.split('.')[0]] = 1/len(tfrfilesNames)
        self.description =  {"inputs": "float", "labels": "float","errWeights":"float"}
        self.nSteps = args['timeSteps']
        self.nInputs = 192
        self.nClass = 32
        self.batchsize = args['batchSize']
        self.whiteNoiseSD = args['whiteNoiseSD']
        self.constantOffsetSD = args['constantOffsetSD']
        self.randomWalkSD = args['randomWalkSD']
        self.smoothInputs = args['smoothInputs']
        self.rnnBinSize = args['rnnBinSize']

    def _shape_add_noise(self,features):
        ## transformations done for the data: reshape and add noise to input
        features["inputs"] = np.reshape(features["inputs"],(self.nSteps, self.nInputs))
        features["labels"] = np.reshape(features["labels"],(self.nSteps, self.nClass))
        noise = np.random.normal(0, self.whiteNoiseSD, size=(self.nSteps,self.nInputs))
        if self.constantOffsetSD > 0 or self.randomWalkSD > 0:
            trainNoise_mn = np.random.normal(0, self.constantOffsetSD, size = (1,self.nInputs))
            trainNoise_mn = np.tile(trainNoise_mn, (self.nSteps, 1))           
            trainNoise_mn += np.cumsum(np.random.normal(0, randomWalkSD, size=(self.nSteps, self.nInputs)), axis=1)
            noise += trainNoise_mn
  
        features["inputs"] += noise
        return features

    def makeDataSet(self):
        dataset = MultiTFRecordDataset(self.tfrecord_pattern, index_pattern=self.index_pattern, splits=self.splits,\
                                      description = self.description,\
                                      transform=self._shape_add_noise,infinite=False,shuffle_queue_size=256)

        return dataset



class handBCI_Dataset(Dataset):
    """hanBCI real dataset per day"""

    def __init__(self, args, inputs, targets, errWeight, numBinsPerTrial, addNoise=True):
        """
        Args:
           
        """
        self.args = args
        self.addNoise = addNoise
        self.nSteps = args['timeSteps']
        self.nInputs = 192
        self.nClass = 32
        self.inputs = inputs
        self.targets = targets 
        self.errWeight = errWeight
        self.numBinsPerTrial = numBinsPerTrial

        self.whiteNoiseSD = args['whiteNoiseSD']
        self.constantOffsetSD = args['constantOffsetSD']
        self.randomWalkSD = args['randomWalkSD']
        self.rnnBinSize = args['rnnBinSize'] 

    def extractSentenceSnippet(self, j, inputs, targets, errWeight, numBinsPerTrial, nSteps, directionality):

        randomStart = np.random.randint(low = 0, high = np.max((numBinsPerTrial[j]+(nSteps-100)-400, 1)))[0]
        # print('snippet range: ',randomStart, randomStart+nSteps)

        inputsSnippet = np.squeeze(inputs[j,randomStart:(randomStart+nSteps),:])
        targetsSnippet = np.squeeze(targets[j,randomStart:(randomStart+nSteps),:])
        charStarts = np.where(targetsSnippet[1:,-1] - targetsSnippet[0:-1,-1]>=0.1)[0]
        def noLetters():
            ews =  np.zeros((nSteps,))
            return ews

        def atLeastOneLetter():
            firstChar = np.cast['int32'](charStarts[0])
            lastChar = np.cast['int32'](charStarts[-1])

            if directionality=='unidirectional':
                #if uni-directional, only need to blank out the first part because it's causal with a delay
                ews =  np.concatenate((np.zeros((firstChar,)), 
                                  errWeight[j,(randomStart+firstChar):(randomStart+nSteps)]), axis=0)
            else:
                #if bi-directional (acausal), we need to blank out the last incomplete character as well so that only fully complete
                #characters are included
                ews =  np.concatenate((np.zeros((firstChar,)), 
                                  errWeight[j, (randomStart+firstChar):(randomStart+lastChar)],
                                  np.zeros((nSteps-lastChar,1))), axis=0)            
            return ews
        if len(charStarts)==0:
            errWeightSnippet = noLetters()
        else:
            errWeightSnippet = atLeastOneLetter()
        # print('shape:',inputsSnippet.shape, targetsSnippet.shape,errWeightSnippet.shape, numBinsPerTrial.shape)
        return inputsSnippet, targetsSnippet, errWeightSnippet, numBinsPerTrial
    
    def _add_noise(self,inputs):
        ## add noise to input
        noise = np.random.normal(0, self.whiteNoiseSD, size=(self.nSteps,self.nInputs))
        if self.constantOffsetSD > 0 or self.randomWalkSD > 0:
            trainNoise_mn = np.random.normal(0, self.constantOffsetSD, size = (1,self.nInputs))
            trainNoise_mn = np.tile(trainNoise_mn, (self.nSteps, 1)) 
            trainNoise_mn += np.cumsum(np.random.normal(0, self.randomWalkSD, size=(self.nSteps, self.nInputs)), axis=1)
            noise += trainNoise_mn
        inputs += noise
        return inputs
    
    def __len__(self):
        return 1024 ## placeholder, large enough compared to batchsize

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        setenceIdx = np.mod(idx, self.inputs.shape[0])
        inputs, labels, ew, _ = self.extractSentenceSnippet(setenceIdx, self.inputs, self.targets,\
                                                            self.errWeight, self.numBinsPerTrial,\
                                self.nSteps, self.args['directionality'])            
        if self.addNoise:
            inputs = self._add_noise(inputs)

        sample = {'inputs': inputs, 'labels': labels,'errWeights': ew} #,'numBinsPerTrial': nbpt
        return sample