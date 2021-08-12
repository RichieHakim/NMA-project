from datetime import datetime
def getDefaultRNNArgs(rootDir, cvPart, outputDir):
    """
    Makes a default 'args' dictionary with all RNN hyperparameters populated with default values.
    """
    args = {}

    #These arguments define each dataset that will be used for training.
    dataDirs = ['t5.2019.05.08']

    for x in range(len(dataDirs)):
        args['timeSeriesFile_'+str(x)] = rootDir+'Step2_HMMLabels/'+cvPart+'/'+dataDirs[x]+'_timeSeriesLabels.mat'
        #args['syntheticDatasetDir_'+str(x)] = rootDir+'Step3_SyntheticSentences/'+cvPart+'/'+dataDirs[x]+'_syntheticSentences/'
        args['cvPartitionFile_'+str(x)] = rootDir+'trainTestPartitions_'+cvPart+'.mat'
        args['sessionName_'+str(x)] = dataDirs[x]

    #Specify which GPU to use (on multi-gpu machines, this prevents tensorflow from taking over all GPUs)
    args['gpuNumber'] = '0'
    
    #mode can either be 'train' or 'inference'
    args['mode'] = 'train'
    
    #where to save the RNN files
    args['outputDir'] = outputDir+'RNNTrainingSteps/'+cvPart
    
    #We can load the variables from a previous run, either to resume training (if loadDir==outputDir) 
    #or otherwise to complete an entirely new training run. 'loadCheckpointIdx' specifies which checkpoint to load (-1 = latest)
    args['loadDir'] = 'None'
    args['loadCheckpointIdx'] = -1
    args['ForTestingOnly'] = False
    #number of units in each GRU layer
    args['nUnits'] = 512
    
    #Specifies how many 10 ms time steps to combine a single bin for RNN processing                              
    args['rnnBinSize'] = 2
    
    #Applies Gaussian smoothing if equal to 1                             
    args['smoothInputs'] = 1
    
    #For the top GRU layer, how many bins to skip for each update (the top layer runs at a slower frequency)                             
    args['skipLen'] = 5
    
    #How many bins to delay the output. Some delay is needed in order to give the RNN enough time to see the entire character
    #before deciding on its identity. Default is 1 second (50 bins).
    args['outputDelay'] = 50 
    
    #Can be 'unidrectional' (causal) or 'bidirectional' (acausal)                              
    args['directionality'] = 'unidirectional'

    #standard deivation of the constant-offset firing rate drift noise                             
    args['constantOffsetSD'] = 0.6
    
    #standard deviation of the random walk firing rate drift noise                             
    args['randomWalkSD'] = 0.02 
   
    #standard deivation of the white noise added to the inputs during training                            
    args['whiteNoiseSD'] = 1.2
    
    #l2 regularization cost                             
    args['l2scale'] = 1e-5 
                                
    args['learnRateStart'] = 0.01
    args['learnRateEnd'] = 0.0
    
    #can optionally specify for only the input layers to train or only the back end                             
    args['trainableInput'] = 1
    args['trainableBackEnd'] = 1

    #this seed is set for numpy and tensorflow when the class is initialized                             
    args['seed'] = datetime.now().microsecond

    #number of checkpoints to keep saved during training                             
    args['nCheckToKeep'] = 1
    
    #how often to save performance statistics                              
    args['batchesPerSave'] = 200 
                                 
    #how often to run a validation diagnostic batch                              
    args['batchesPerVal'] = 50 
                                 
    #how often to save the model                             
    args['batchesPerModelSave'] = 5000 
                                 
    #how many minibatches to use total                             
    args['nBatchesToTrain'] = 100000 

    #number of time steps to use in the minibatch (1200 = 24 seconds)                             
    args['timeSteps'] = 1200 
                                 
    #number of sentence snippets to include in the minibatch 
    args['batchSize'] = 16 ## 64 for GPU with larger memory
                                 
    #how much of each minibatch is synthetic data                              
    args['synthBatchSize'] = 0 ## set to zero to skip synthetic data.

    #can be used to scale up all input features, sometimes useful when transferring to new days without retraining 
    args['inputScale'] = 1.0
                                 
    #parameters to specify where to save the outputs and which layer to use during inference                             
    args['inferenceOutputFileName'] = 'None'
    args['inferenceInputLayer'] = 0

    #defines the mapping between each day and which input layer to use for that day                             
    args['dayToLayerMap'] = '[0]' ## redefined below
                                 
    #for each day, the probability that a minibatch will pull from that day. Can be used to weight some days more than others  
    args['dayProbability'] = '[1.0]'

    return args