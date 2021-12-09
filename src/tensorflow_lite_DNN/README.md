Code in Tensorflow to train a DNN using our data and convert to tflite

# Requirements and Setup

First, create a system variable called 'SMART_TOOLS_ROOT_DIR' that points to where
you downloaded the repository.

On Linux, for example, make your bashrc have:

`export SMART_TOOLS_ROOT_DIR="{path to your repo}"`

Same for the lab plotting utils on GIT. An example for my machine in my bashrc:

`export SMART_TOOLS_ROOT_DIR='/Users/spc2294/Documents/work/code/SmartTools/SmartTools'`

`export UTILS_DIR='/Users/spc2294/Documents/work/UT_work/lab_resources/plotting_utils/'`

For managing software, consider using Anaconda.

## IMPORTANT: How to use the correct Anaconda Env for this code

We have two conda env's stored in this repo, which are ready to use. 

Pytorch: `SMART_TOOLS_ROOT_DIR/requirements/pytorch_conda_requirements.txt`

Tensorflow: `SMART_TOOLS_ROOT_DIR/requirements/tensorflow_conda_requirements.txt`

As stated at the top of these files, create your env using:

`conda create --name <env> --file <this file>`

For example:
`conda create --name tf --file tensorflow_conda_requirements.txt`

# Code Structure

### First, train the tf activity classifier
`python3 -i tf_train_activity_classifier.py`

### This will use training utils stored in
`utils_tensorflow.py`

### Finally, plot loss
`plot_loss.py`

This command will store plots and the trained model in `${SMART_TOOLS_ROOT_DIR}/scratch/tensorflow_classifier/`

# Saved Models and Model Statistics

Models are saved at `${SMART_TOOLS_ROOT_DIR}/`

## Original Tensorflow Model

    Total params: 59,028

    Trainable params: 59,028

    Model size: 230.578125 KB


## Tensorflow Lite Model
	
    Basic model is 234 Kilobytes

    Quantized model is 63 Kilobytes

    Difference is 170 Kilobytes

# Example Code from Magic Wand

    - from: https://github.com/tensorflow/tflite-micro/blob/main/tensorflow/lite/micro/examples/magic_wand/train/train.py
    - good link:
    - https://stackoverflow.com/questions/58576446/error-in-keras-digit-dataset-expected-conv2d-input-to-have-4-dimensions-but-go

# Extremely Important: How to Scale Data Systematically from Cloud to Arduino

- example blog: `https://machinelearningmastery.com/how-to-save-and-load-models-and-data-preparation-in-scikit-learn-for-later-use/`

