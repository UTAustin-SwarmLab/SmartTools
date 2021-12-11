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

### Next, plot loss
`plot_loss.py`

### KEY STEP: TEST the quantized models and write results to a file
`python3 -i test_pretrained_model_and_scaling.py`

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

- we do not use scikit learn on purpose (unlike Pytorch)
- instead, we save the quantiles of each column in the training data to a csv
- then, for any new dataset we scale by the .001 and .999 quantile of each column, which we can easily do in C

- `normalized_x = (x - quantile_.001[x]) / (quantile_.999[x] - quantile_.001[x])`

# List of Sensors and Features Expected in Order
- 110 variable inputs
- 11 sensors x 10 features
- full list: 

-  `['accX_min', 'accX_max', 'accX_mean', 'accX_kurt', 'accX_sem', 'accX_std', 'accX_var', 'accX_skew', 'accX_mad', 'accX_sum', 'accY_min', 'accY_max', 'accY_mean', 'accY_kurt', 'accY_sem', 'accY_std', 'accY_var', 'accY_skew', 'accY_mad', 'accY_sum', 'accZ_min', 'accZ_max', 'accZ_mean', 'accZ_kurt', 'accZ_sem', 'accZ_std', 'accZ_var', 'accZ_skew', 'accZ_mad', 'accZ_sum', 'wx_min', 'wx_max', 'wx_mean', 'wx_kurt', 'wx_sem', 'wx_std', 'wx_var', 'wx_skew', 'wx_mad', 'wx_sum', 'wy_min',
    'wy_max', 'wy_mean', 'wy_kurt', 'wy_sem', 'wy_std', 'wy_var', 'wy_skew', 'wy_mad', 'wy_sum', 'wz_min', 'wz_max', 'wz_mean', 'wz_kurt', 'wz_sem', 'wz_std', 'wz_var', 'wz_skew', 'wz_mad', 'wz_sum', 'bx_min', 'bx_max', 'bx_mean', 'bx_kurt', 'bx_sem', 'bx_std', 'bx_var', 'bx_skew', 'bx_mad', 'bx_sum', 'by_min', 'by_max', 'by_mean', 'by_kurt', 'by_sem', 'by_std', 'by_var', 'by_skew', 'by_mad', 'by_sum', 'bz_min', 'bz_max', 'bz_mean', 'bz_kurt', 'bz_sem', 'bz_std', 'bz_var', 'bz_skew', 'bz_mad',
    'bz_sum', 'Isens_min', 'Isens_max', 'Isens_mean', 'Isens_kurt', 'Isens_sem', 'Isens_std', 'Isens_var', 'Isens_skew', 'Isens_mad', 'Isens_sum', 'Srms_min', 'Srms_max', 'Srms_mean', 'Srms_kurt', 'Srms_sem', 'Srms_std', 'Srms_var', 'Srms_skew', 'Srms_mad', 'Srms_sum']`

- Sensors: `[accX, accY, accZ, wx, wy, wz, bx, by, bz, Isens, Srms]`
- Features: `[min, max, mean, kurt, sem, std, var, skew, mad, sum]`
- Data Input to DNN: 
    - `11 x 10` NORMALIZED matrix  
