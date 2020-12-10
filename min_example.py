import tensorflow as tf

mnist = tf.keras.datasets.mnist

# X = data
# Y = labels
# train = training data (80%)
# test = testing data (20%)
(x_train, y_train), (x_test, y_test) = mnist.load_data()

def normalize(x_train, x_test):
  range = 255.0
  return x_train / range, x_test / range

x_train, x_test = normalize(x_train, x_test)
