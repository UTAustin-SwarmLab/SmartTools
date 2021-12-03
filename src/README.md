
# Requirements and Setup

First, create a system variable called 'SMART_TOOLS_ROOT_DIR' that points to where
you downloaded the repository.

On Linux, for example, make your bashrc have:

`export SMART_TOOLS_ROOT_DIR="{path to your repo}"`

Same for the lab plotting utils on GIT. An example for my machine in my bashrc:

`export SMART_TOOLS_ROOT_DIR='/Users/spc2294/Documents/work/code/SmartTools/SmartTools'`

`export UTILS_DIR='/Users/spc2294/Documents/work/UT_work/lab_resources/plotting_utils/'`

For managing software, consider using Anaconda.

# Code Structure

### First, run the random forest
`python3 -i simple_random_forest_classifier.py`

### Next, run the DNN in Pytorch

`python3 -i basic_neural_network_classifier.py`

This command will store plots and the trained model in `${SMART_TOOLS_ROOT_DIR}/scratch`


