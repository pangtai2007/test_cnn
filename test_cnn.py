__author__ = 'chapter'

import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

def variable_summaries(var):
    """Attach a lot of summaries to a Tensor (for TensorBoard visualization)."""
    with tf.name_scope('summaries'):
      mean = tf.reduce_mean(var)
      tf.summary.scalar('mean', mean)
      with tf.name_scope('stddev'):
        stddev = tf.sqrt(tf.reduce_mean(tf.square(var - mean)))
      tf.summary.scalar('stddev', stddev)
      tf.summary.scalar('max', tf.reduce_max(var))
      tf.summary.scalar('min', tf.reduce_min(var))
      tf.summary.histogram('histogram', var)

def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)

def bias_variable(shape):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)

def conv2d(x, W):
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

def max_pool_2x2(x):
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

def fc_layer(input_tensor, input_dim, output_dim, layer_name, act=tf.nn.relu):
  """Reusable code for making a simple neural net layer.

  It does a matrix multiply, bias add, and then uses relu to nonlinearize.
  It also sets up name scoping so that the resultant graph is easy to read,
  and adds a number of summary ops.
  """
  # Adding a name scope ensures logical grouping of the layers in the graph.
  with tf.name_scope(layer_name):
    # This Variable will hold the state of the weights for the layer
    with tf.name_scope('weights'):
      weights = weight_variable([input_dim, output_dim])
      variable_summaries(weights)
    with tf.name_scope('biases'):
      biases = bias_variable([output_dim])
      variable_summaries(biases)
    with tf.name_scope('Wx_plus_b'):
      preactivate = tf.matmul(input_tensor, weights) + biases
      tf.summary.histogram('pre_activations', preactivate)
    activations = act(preactivate, name='activation')
    tf.summary.histogram('activations', activations)
    return activations

def conv_layer(input_tensor, input_deep, conv_size, conv_deep, layer_name, act=tf.nn.relu):
  # need to show conv_layer in tensorboard
  # I want to visualize conv_layer in the function, including a conv and a pooling layer
  with tf.name_scope(layer_name):
    weights = weight_variable([conv_size, conv_size, input_deep, conv_deep])
    bias = bias_variable([conv_deep])
    conv = act(conv2d(input_tensor, weights)+bias)
    # need to add visualization
    return conv

def pool_layer(conv, layer_name):
  with tf.name_scope(layer_name):
    pool = max_pool_2x2(conv)
    # need to add visualization
    return pool


mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)
print("Download Done!")

sess = tf.InteractiveSession()

# paras
#W_conv1 = weight_variable([5, 5, 1, 32])
#b_conv1 = bias_variable([32])

# conv layer-1
x = tf.placeholder(tf.float32, [None, 784])
x_image = tf.reshape(x, [-1, 28, 28, 1])

# change
conv1 = conv_layer(x_image, 1, 5, 32, 'conv1')
pool1 = pool_layer(conv1, 'pool1')
#h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)
#h_pool1 = max_pool_2x2(h_conv1)

# conv layer-2
conv2 = conv_layer(conv1, 32, 5, 64, 'conv1')
pool2 = pool_layer(conv2, 'pool2')

#W_conv2 = weight_variable([5, 5, 32, 64])
#b_conv2 = bias_variable([64])

#h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)
#h_pool2 = max_pool_2x2(h_conv2)

# full connection
#W_fc1 = weight_variable([7 * 7 * 64, 1024])
#b_fc1 = bias_variable([1024])

pool2_flat = tf.reshape(pool2, [-1, 7 * 7 * 64])
fc1 = fc_layer(pool2_flat, 7 * 7 * 64, 1024, 'fc1')
#h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)

# dropout
with tf.name_scope('dropout'):
  keep_prob = tf.placeholder(tf.float32)
  fc1_drop = tf.nn.dropout(fc1, keep_prob)

# output layer: softmax
#W_fc2 = weight_variable([1024, 10])
#b_fc2 = bias_variable([10])

#y_conv = tf.nn.softmax(tf.matmul(h_fc1_drop, W_fc2) + b_fc2)
y = fc_layer(fc1, 1024, 10, 'softmax', act=tf.nn.softmax)
y_ = tf.placeholder(tf.float32, [None, 10])

# model training
cross_entropy = -tf.reduce_sum(y_ * tf.log(y))
train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)

correct_prediction = tf.equal(tf.arg_max(y, 1), tf.arg_max(y_, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

sess.run(tf.initialize_all_variables())

for i in range(20000):
    batch = mnist.train.next_batch(50)

    if i % 100 == 0:
        #train_accuacy = accuracy.eval(feed_dict={x: batch[0], y_: batch[1], keep_prob: 1.0})
        #print("step %d, training accuracy %g"%(i, train_accuacy))
        print('lll')
        print(sess.run(y, feed_dict = {x: batch[0], keep_prob: 0.5}))
    train_step.run(feed_dict = {x: batch[0], y_: batch[1], keep_prob: 0.5})

# accuacy on test
print("test accuracy %g"%(accuracy.eval(feed_dict={x: mnist.test.images, y_: mnist.test.labels, keep_prob: 1.0})))
