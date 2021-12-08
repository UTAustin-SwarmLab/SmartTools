import tensorflow as tf

def reshape_function(data, label, num_sensors=11, num_features=10):
  reshaped_data = tf.reshape(data, [-1, num_sensors, num_features])
  return reshaped_data, label

def calculate_model_size(model):
  print(model.summary())
  var_sizes = [
      np.product(list(map(int, v.shape))) * v.dtype.size
      for v in model.trainable_variables
  ]
  print("Model size:", sum(var_sizes) / 1024, "KB")


def build_cnn(seq_length):
  """Builds a convolutional neural network in Keras."""
  model = tf.keras.Sequential([
      tf.keras.layers.Conv2D(
          8, (4, 3),
          padding="same",
          activation="relu",
          input_shape=(seq_length, 3, 1)),  # output_shape=(batch, 128, 3, 8)
      tf.keras.layers.MaxPool2D((3, 3)),  # (batch, 42, 1, 8)
      tf.keras.layers.Dropout(0.1),  # (batch, 42, 1, 8)
      tf.keras.layers.Conv2D(16, (4, 1), padding="same",
                             activation="relu"),  # (batch, 42, 1, 16)
      tf.keras.layers.MaxPool2D((3, 1), padding="same"),  # (batch, 14, 1, 16)
      tf.keras.layers.Dropout(0.1),  # (batch, 14, 1, 16)
      tf.keras.layers.Flatten(),  # (batch, 224)
      tf.keras.layers.Dense(16, activation="relu"),  # (batch, 16)
      tf.keras.layers.Dropout(0.1),  # (batch, 16)
      tf.keras.layers.Dense(4, activation="softmax")  # (batch, 4)
  ])
  model_path = os.path.join("./netmodels", "CNN")
  print("Built CNN.")
  if not os.path.exists(model_path):
    os.makedirs(model_path)
  model.load_weights("./netmodels/CNN/weights.h5")
  return model, model_path

