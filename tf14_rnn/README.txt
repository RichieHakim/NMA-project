This is based on the original code https://github.com/fwillett/handwritingBCI

## Working with Tensorflow 1.14.0. Original model may user lower version than this.
Currently, TF 1.14 comes with gast 0.5.1 which compatability issue. 
Downgrading gast to lower version: gast==0.2.2 per https://github.com/tensorflow/tensorflow/issues/32949
Other dependency: google-pasta==0.1.6

## Folder structure
under "working " there are two folders for storing training checkpoints and inference files respectively

## Use the "DataSet" folder for data as original repo.

## Main files
RNN network: tf14_RNNi.py
Hyperparameters: handleDefaultParameters.py
Training: tr14_rnnTraning.py
Inference: tf14_rnnInferencei.py
other files are copies from the original repo.
