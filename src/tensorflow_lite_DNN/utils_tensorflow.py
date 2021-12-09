import tensorflow as tf
import pandas
import sys,os
from collections import OrderedDict
import argparse
import numpy as np
import datetime

from plotting_utils import *
from textfile_utils import *
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import tensorflow as tf
from utils_tensorflow import *

def calculate_model_size(model):
  print(model.summary())
  var_sizes = [
      np.product(list(map(int, v.shape))) * v.dtype.size
      for v in model.trainable_variables
  ]
  model_size = sum(var_sizes) / 1024

  print("Model size:", model_size, "KB")
  return model_size


def build_1D_CNN(model_base_dir, model_name = '1DCNN', num_sensors = 11, num_features = 10, num_outputs = 4):

  """Builds a convolutional neural network in Keras."""
  model = tf.keras.Sequential([
        tf.keras.layers.Conv1D(56, 3, activation='relu', input_shape=(num_sensors, num_features)),
        tf.keras.layers.MaxPooling1D(2, 2),
        tf.keras.layers.Conv1D(56, 3, activation='relu'),
        tf.keras.layers.MaxPooling1D(2, 2),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(784, activation='relu'),
        tf.keras.layers.Dense(num_outputs, activation='softmax')])

  model_path = None

  #model_path = os.path.join(model_base_dir + "./netmodels", "1DCNN")
  #print("Built CNN.")
  #if not os.path.exists(model_path):
  #  os.makedirs(model_path)
  #model.load_weights(model_base_dir + "./netmodels/1DCNN/weights.h5")

  return model, model_path


def train_net(
    model,
    model_path,  # pylint: disable=unused-argument
    train_len,  # pylint: disable=unused-argument
    train_data,
    valid_len,
    valid_data,  # pylint: disable=unused-argument
    test_len,
    test_data,
    kind):
  """Trains the model."""
  calculate_model_size(model)
  epochs = 50
  batch_size = 64
  model.compile(optimizer="adam",
                loss="sparse_categorical_crossentropy",
                metrics=["accuracy"])
  if kind == "CNN":
    train_data = train_data.map(reshape_function)
    test_data = test_data.map(reshape_function)
    valid_data = valid_data.map(reshape_function)
  test_labels = np.zeros(test_len)
  idx = 0
  for data, label in test_data:  # pylint: disable=unused-variable
    test_labels[idx] = label.numpy()
    idx += 1
  train_data = train_data.batch(batch_size).repeat()
  valid_data = valid_data.batch(batch_size)
  test_data = test_data.batch(batch_size)
  model.fit(train_data,
            epochs=epochs,
            validation_data=valid_data,
            steps_per_epoch=1000,
            validation_steps=int((valid_len - 1) / batch_size + 1),
            callbacks=[tensorboard_callback])
  loss, acc = model.evaluate(test_data)
  pred = np.argmax(model.predict(test_data), axis=1)
  confusion = tf.math.confusion_matrix(labels=tf.constant(test_labels),
                                       predictions=tf.constant(pred),
                                       num_classes=4)
  print(confusion)
  print("Loss {}, Accuracy {}".format(loss, acc))

  # Convert the model to the TensorFlow Lite format without quantization
  converter = tf.lite.TFLiteConverter.from_keras_model(model)
  tflite_model = converter.convert()

  # Save the model to disk
  open("model.tflite", "wb").write(tflite_model)

  # Convert the model to the TensorFlow Lite format with quantization
  converter = tf.lite.TFLiteConverter.from_keras_model(model)
  converter.optimizations = [tf.lite.Optimize.DEFAULT]
  tflite_model = converter.convert()

  # Save the model to disk
  open("model_quantized.tflite", "wb").write(tflite_model)

  basic_model_size = os.path.getsize("model.tflite")
  print("Basic model is %d bytes" % basic_model_size)
  quantized_model_size = os.path.getsize("model_quantized.tflite")
  print("Quantized model is %d bytes" % quantized_model_size)
  difference = basic_model_size - quantized_model_size
  print("Difference is %d bytes" % difference)
