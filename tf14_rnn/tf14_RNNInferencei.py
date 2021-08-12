#!/usr/bin/env python
# coding: utf-8

# In[ ]:


#This notebook takes a previously trained RNN and evaluates it on held-out data, saving the outputs for later processing.
#We also compute here the overall character error rate and word error rate across all held-out data. 


# In[ ]:


import tensorflow as tf
tf.compat.v1.reset_default_graph()
#suppress all tensorflow warnings (largely related to compatability with v2)
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

import numpy as np
import scipy.io
import os
import matplotlib.pyplot as plt
from datetime import datetime
from tf14_RNNi import charSeqRNN
from handleDefaultParameters import getDefaultRNNArgs
#point this towards the top level dataset directory
rootDir = '../handwritingBCIData/'
outDir = 'working/'
#evaluate the RNN on these datasets
dataDirs = ['t5.2019.05.08','t5.2019.11.25','t5.2019.12.09','t5.2019.12.11','t5.2019.12.18',
            't5.2019.12.20','t5.2020.01.06','t5.2020.01.08','t5.2020.01.13','t5.2020.01.15']

#use this train/test partition
cvPart = 'HeldOutTrials'

#point this towards the specific RNN we want to evaluate
rnnOutputDir = cvPart

#this prevents tensorflow from taking over more than one gpu on a multi-gpu machine
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]='0'

#this is where we're going to save the RNN outputs
inferenceSaveDir = outDir+'RNNInference/' + rnnOutputDir

#all RNN runs are saved in 'Step4_RNNTraining'
if not os.path.isdir(outDir + 'RNNInference'):
    os.mkdir(outDir + 'RNNInference')
if not os.path.isdir(outDir + 'RNNInference/'+rnnOutputDir):
    os.mkdir(outDir + 'RNNInference/'+rnnOutputDir)


# In[ ]:


#Configures the RNN for inference mode.
#args = getDefaultRNNArgs()
args = getDefaultRNNArgs(rootDir, cvPart, outDir)
args['outputDir'] = outDir + 'RNNTrainingSteps/'+rnnOutputDir
args['loadDir'] = args['outputDir']
args['mode'] = 'infer'
args['timeSteps'] = 7500 #Need to specify enough time steps so that the longest sentence fits in the minibatch
args['batchSize'] = 2 #Process just two sentences at a time, to make sure we have enough memory
args['synthBatchSize'] = 0 #turn off synthetic data here, we are only using real data

#Proceeds one dataset at a time. Currently the code is setup to only process a single dataset at inference time,
#so we have to rebuild the graph for each dataset.
for x in range(len(dataDirs)):
    #configure the RNN to process this particular dataset
    print(' ')
    print('Processing dataset ' + dataDirs[x])
    
    args['sentencesFile_0'] = rootDir+'Datasets/'+dataDirs[x]+'/sentences.mat'
    args['singleLettersFile_0'] = rootDir+'Datasets/'+dataDirs[x]+'/singleLetters.mat'
    args['labelsFile_0'] = rootDir+'RNNTrainingSteps/Step2_HMMLabels/'+cvPart+'/'+dataDirs[x]+'_timeSeriesLabels.mat'
    args['syntheticDatasetDir_0'] = rootDir+'RNNTrainingSteps/Step3_SyntheticSentences/'+cvPart+'/'+dataDirs[x]+'_syntheticSentences/'
    args['cvPartitionFile_0'] = rootDir+'RNNTrainingSteps/trainTestPartitions_'+cvPart+'.mat'
    args['sessionName_0'] = dataDirs[x]

    args['inferenceOutputFileName'] = inferenceSaveDir + '/' + dataDirs[x] + '_inferenceOutputs.mat'
    args['inferenceInputLayer'] = x
    
    #instantiate the RNN model
    rnnModel = charSeqRNN(args=args)

    #evaluate the RNN on the held-out data
    outputs = rnnModel.inference()
    
    #reset the graph to make space for the next dataset
    tf.reset_default_graph()






