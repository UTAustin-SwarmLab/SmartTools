'''
    example of how to run the pre-trained DNN
'''

import sys,os
import numpy as np
import tensorflow as tf

# data is expected as:
# BATCH_SIZE x NUM_SENSORS x NUM_FEATURES
# view this as an image with 1 channel, NUM_SENSORS x NUM_FEATURES size

# for a single inference, we have a batch size of 1

if __name__ == '__main__':

    # first load the pre-trained CNN model
    base_dir = SMART_TOOLS_ROOT_DIR + '/pretrained_models/tensorflow_classifier/'
    model_base_dir = base_dir + '/tf_model/'

    tflite_quantized_model_path = model_base_dir + '/model_quantized.tflite'
    tflite_model_path = model_base_dir + '/model.tflite'

    # compare both our quantized and float models
    model_type_list = [('TFLite Quantized', tflite_quantized_model_path), ('TFLite Float', tflite_model_path)]

    for model_type, model_path in model_type_list:

        # actually load the tflite model from disk
        interpreter, input_details, output_details, input_shape = load_trained_tflite_model(model_path)

		for i in NUM_TRIALS:

			# load in random data of the correct size of 1 x NUM_SENSORS x NUM_FEATURES
			# we expect this is properly normalized!
			input_data = np.array(np.random.random_sample(input_shape), dtype=np.float32)

			# actually run the DNN
			interpreter.set_tensor(input_details[0]['index'], input_data)
			interpreter.invoke()

			# get the tensor outputs (probability per class)
			output_data = interpreter.get_tensor(output_details[0]['index'])

			# get the most likely prediction, this is a number between 0 and NUM_CLASSES-1
			preds = np.squeeze(output_data)
			argmax_pred = np.argmax(preds)

			print(' ')
			print('trial i: ', i)
			print('preds: ', preds, np.sum(preds))
			print('argmax_pred: ', argmax_pred)
			print(' ')

