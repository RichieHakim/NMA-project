import numpy as np
import os
from datetime import datetime
from datetime import datetime
import tensorflow as tf 
import random
from dataPreprocessing import prepareDataCubesForRNN
import sys
import scipy.io
from scipy.ndimage.filters import gaussian_filter1d
import scipy.special
from handleDefaultParameters import getDefaultRNNArgs
from tf14_RNNi import charSeqRNN ## modified from original to be tf 1.14 compatible RNN
#point this towards the top level dataset directory
rootDir = '../handwritingBCIData/'
outDir = 'working/'
#train an RNN using data from these specified sessions
dataDirs = ['t5.2019.05.08','t5.2019.11.25','t5.2019.12.09','t5.2019.12.11','t5.2019.12.18',
            't5.2019.12.20','t5.2020.01.06','t5.2020.01.08','t5.2020.01.13','t5.2020.01.15']

#print([x.split('_')[1] for x in list(slDat.keys()) if x.startswith('neuralActivityCube')])
## from singleletter data
# charatersUsed = ['a', 'b', 'c', 'd', 't', 'm', 'o', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', \
#      'n', 'p', 'q', 'r', 's', 'u', 'v', 'w', 'x', 'y', 'z', \
#          'greaterThan', 'comma', 'apostrophe', 'tilde', 'questionMark']
#use this train/test partition 
cvPart = 'HeldOutTrials'

#name of the directory where this RNN run will be saved
rnnOutputDir = cvPart

#all RNN runs are saved in 'Step4_RNNTraining'
if not os.path.isdir(outDir + 'RNNTrainingSteps'):
    os.mkdir(outDir + 'RNNTrainingSteps')
if not os.path.isdir(outDir + 'RNNTrainingSteps/'+rnnOutputDir):
    os.mkdir(outDir + 'RNNTrainingSteps/'+rnnOutputDir)

args = getDefaultRNNArgs(rootDir, cvPart, outDir)
#Configure the arguments for a multi-day RNN (that will have a unique input layer for each day)
for x in range(len(dataDirs)):
    args['sentencesFile_'+str(x)] = rootDir+'Datasets/'+dataDirs[x]+'/sentences.mat'
    args['singleLettersFile_'+str(x)] = rootDir+'Datasets/'+dataDirs[x]+'/singleLetters.mat'
    args['labelsFile_'+str(x)] = rootDir+'RNNTrainingSteps/Step2_HMMLabels/'+cvPart+'/'+dataDirs[x]+'_timeSeriesLabels.mat'
    args['syntheticDatasetDir_'+str(x)] = rootDir+'RNNTrainingSteps/Step3_SyntheticSentences/'+cvPart+'/'+dataDirs[x]+'_syntheticSentences/'
    args['cvPartitionFile_'+str(x)] = rootDir+'RNNTrainingSteps/trainTestPartitions_'+cvPart+'.mat'
    args['sessionName_'+str(x)] = dataDirs[x]
    
if not os.path.isdir(args['outputDir']):
    os.mkdir(args['outputDir'])
    
#this weights each day equally (0.1 probability for each day) and allocates a unique input layer for each day (0-9)
args['dayProbability'] = '[0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1]'
args['dayToLayerMap'] = '[0,1,2,3,4,5,6,7,8,9]'
# args['verbose'] = True ## extra print-out information

#instantiate the RNN model
args['mode'] = 'train' ## make sure it is set in 'train' mode
args['ForTestingOnly'] = False ## FOR DEBUGING. set "self.nDays = 2" (use 2 days of data for testing run)
rnnModel = charSeqRNN(args=args)
#train 
rnnModel.train()

## traing and validation error and accurcey
## 1000 minibatches with batch size 16
# Val Batch: 970/1000, valErr: 2.633857, trainErr: 2.5772016, Val Acc.: 0.20126879, Train Acc.: 0.16488159, grad: 0.88078076, learnRate: 0.0003000000000000003, time: 0.125
# Val Batch: 980/1000, valErr: 2.170796, trainErr: 2.7036781, Val Acc.: 0.112816304, Train Acc.: 0.16973126, grad: 0.924166, learnRate: 0.00020000000000000017, time: 0.140623
# Val Batch: 990/1000, valErr: 2.6113248, trainErr: 2.831241, Val Acc.: 0.18956044, Train Acc.: 0.18609758, grad: 0.86157906, learnRate: 0.00010000000000000009, time: 0.140626

# use the other notebok :'tf14_RNNInference.py' for inference
