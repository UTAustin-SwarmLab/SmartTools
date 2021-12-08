import tensorflow as tf

def reshape_function(data, label, num_sensors=11, num_features=10):
  reshaped_data = tf.reshape(data, [-1, num_sensors, num_features])
  return reshaped_data, label
