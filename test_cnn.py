# cnn mnist tensorboard

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import sys

import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

FLAGS = None

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
  """
  Reusable code for making a simple neural net layer.
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
    with tf.name_scope('weights'):
      weights = weight_variable([conv_size, conv_size, input_deep, conv_deep])
      variable_summaries(weights)
    with tf.name_scope('biases'):
      bias = bias_variable([conv_deep])
      variable_summaries(bias)
    conv = act(conv2d(input_tensor, weights)+bias)
    # need to add visualization
    return conv

def pool_layer(conv, layer_name):
  with tf.name_scope(layer_name):
    with tf.name_scope('pool'):
      pool = max_pool_2x2(conv)
      variable_summaries(pool)
  # need to add visualization
  return pool

def train():
  mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)
  print("Download Done!")

  sess = tf.InteractiveSession()

  # conv layer-1
  with tf.name_scope('input'):
    x = tf.placeholder(tf.float32, [None, 784], name = 'x_input')
    y_ = tf.placeholder(tf.float32, [None, 10], name = 'y_input')
  with tf.name_scope('input_reshape'):
    x_image = tf.reshape(x, [-1, 28, 28, 1], name = 'x_image')
    tf.summary.image('input', x_image, 10)

  # change
  conv1 = conv_layer(x_image, 1, 5, 32, 'conv1')
  pool1 = pool_layer(conv1, 'pool1')

  # conv layer-2
  conv2 = conv_layer(pool1, 32, 5, 64, 'conv2')
  pool2 = pool_layer(conv2, 'pool2')

  pool2_flat = tf.reshape(pool2, [-1, 7 * 7 * 64])
  fc1 = fc_layer(pool2_flat, 7 * 7 * 64, 1024, 'fc1')

  # dropout
  with tf.name_scope('dropout'):
    keep_prob = tf.placeholder(tf.float32)
    fc1_drop = tf.nn.dropout(fc1, keep_prob)

  # output layer: softmax
  y = fc_layer(fc1, 1024, 10, 'softmax', act=tf.nn.softmax)

  # model training
  with tf.name_scope('cross_entropy'):
    cross_entropy = -tf.reduce_sum(y_ * tf.log(y))
    tf.summary.scalar('cross_entropy', cross_entropy)
  with tf.name_scope('train'):
    train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)

  with tf.name_scope('accuracy'):
    correct_prediction = tf.equal(tf.arg_max(y, 1), tf.arg_max(y_, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    tf.summary.scalar('accuracy', accuracy)


  # Merge all the summaries and write them out to /tmp/mnist_logs (by default)
  merged = tf.summary.merge_all()
  train_writer = tf.summary.FileWriter(FLAGS.log_dir + '/train', sess.graph)
  test_writer = tf.summary.FileWriter(FLAGS.log_dir + '/test')
  sess.run(tf.initialize_all_variables())

  for i in range(FLAGS.max_steps):
      batch = mnist.train.next_batch(50)

      if i % 100 == 0:
          run_options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
          run_metadata = tf.RunMetadata()
          summary, _ = sess.run([merged, train_step],
                              feed_dict={x: batch[0], y_: batch[1], keep_prob: 1.0},
                              options=run_options,
                              run_metadata=run_metadata)
          train_writer.add_run_metadata(run_metadata, 'step%03d' % i)
          train_writer.add_summary(summary, i)
          print('Adding run metadata for', i)
          #train_accuacy = accuracy.eval(feed_dict={x: batch[0], y_: batch[1], keep_prob: 1.0})
          #print("step %d, training accuracy %g"%(i, train_accuacy))

      train_step.run(feed_dict = {x: batch[0], y_: batch[1], keep_prob: 1.0})
  train_writer.close()
  test_writer.close()

# accuacy on test
#print("test accuracy %g"%(accuracy.eval(feed_dict={x: mnist.test.images, y_: mnist.test.labels, keep_prob: 1.0})))
def main(_):
  if tf.gfile.Exists(FLAGS.log_dir):
    tf.gfile.DeleteRecursively(FLAGS.log_dir)
  tf.gfile.MakeDirs(FLAGS.log_dir)
  train()


if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument('--max_steps', type=int, default=1000,
                      help='Number of steps to run trainer.')
  parser.add_argument('--log_dir', type=str, default='cnn_mnist_log',
                      help='Summaries log directory')
  FLAGS, unparsed = parser.parse_known_args()
  tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)
