Code in Tensorflow to train a DNN using our data and convert to tflite

## Example Code from Magic Wand

- `example_magic_wand_train.py`
- from: https://github.com/tensorflow/tflite-micro/blob/main/tensorflow/lite/micro/examples/magic_wand/train/train.py

- good link:
- https://stackoverflow.com/questions/58576446/error-in-keras-digit-dataset-expected-conv2d-input-to-have-4-dimensions-but-go

## Sample Run

For our Actual 1D CNN model:


### Original Tensorflow Model

`_________________________________________________________________
Layer (type)                 Output Shape              Param #   
=================================================================
conv1d (Conv1D)              (None, 9, 56)             1736      
_________________________________________________________________
max_pooling1d (MaxPooling1D) (None, 4, 56)             0         
_________________________________________________________________
conv1d_1 (Conv1D)            (None, 2, 56)             9464      
_________________________________________________________________
max_pooling1d_1 (MaxPooling1 (None, 1, 56)             0         
_________________________________________________________________
flatten (Flatten)            (None, 56)                0         
_________________________________________________________________
dense (Dense)                (None, 784)               44688     
_________________________________________________________________
dense_1 (Dense)              (None, 4)                 3140      
=================================================================
Total params: 59,028
Trainable params: 59,028
Non-trainable params: 0
_________________________________________________________________
None
Model size: 230.578125 KB
`

### Tensorflow Lite Model
`
	Basic model is 234 Kilobytes
	Quantized model is 63 Kilobytes
	Difference is 170 Kilobytes
`
