# coding: utf-8

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import _pickle as pickle
import seaborn


def unpickle(filename):
    '''解压数据'''
    with open(filename, 'rb') as f:
        d = pickle.load(f, encoding='latin1')
        return d


def onehot(labels):
    '''one-hot 编码'''
    n_sample = len(labels)
    n_class = max(labels) + 1
    onehot_labels = np.zeros((n_sample, n_class))
    onehot_labels[np.arange(n_sample), labels] = 1
    return onehot_labels


# 训练数据集
data1 = unpickle('cifar10-dataset/data_batch_1')
data2 = unpickle('cifar10-dataset/data_batch_2')
data3 = unpickle('cifar10-dataset/data_batch_3')
data4 = unpickle('cifar10-dataset/data_batch_4')
data5 = unpickle('cifar10-dataset/data_batch_5')
X_train = np.concatenate((data1['data'], data2['data'], data3['data'], data4['data'], data5['data']), axis=0)
y_train = np.concatenate((data1['labels'], data2['labels'], data3['labels'], data4['labels'], data5['labels']), axis=0)
y_train = onehot(y_train)
# 测试数据集
test = unpickle('cifar10-dataset/test_batch')
X_test = test['data'][:5000, :]
y_test = onehot(test['labels'])[:5000, :]
del test

print('Training dataset shape:', X_train.shape)
print('Training labels shape:', y_train.shape)
print('Testing dataset shape:', X_test.shape)
print('Testing labels shape:', y_test.shape)

with tf.device('/cpu:0'):
    
    def nextbatch(train, test, batch_size=128):
        '''获得下一个batch'''
        total_batch = int(train.shape[0] / batch_size)
        idx = np.random.randint(0, total_batch)
        return (train[idx*batch_size : (idx+1)*batch_size, :],
                test[idx*batch_size : (idx+1)*batch_size, :])
    
    # 模型参数
    learning_rate = 1e-3
    training_iters = 100
    batch_size = 50
    display_step = 5
    n_features = 3072  # 32*32*3
    n_classes = 10
    n_fc1 = 384
    n_fc2 = 192

    # 构建模型
    with tf.name_scope('input'):
        x = tf.placeholder(tf.float32, [None, n_features])
        y = tf.placeholder(tf.float32, [None, n_classes])


    def weight_variable(layer, shape, stddev, name):
        with tf.name_scope(layer):
            return tf.Variable(tf.truncated_normal(shape, stddev=stddev, name=name))


    def bias_variable(layer, value, dtype, shape, name):
        with tf.name_scope(layer):
            return tf.Variable(tf.constant(value, dtype=dtype, shape=shape, name=name))


    with tf.name_scope('input_reshape'):
        x4d = tf.reshape(x, [-1, 32, 32, 3])

    # 卷积层 1
    with tf.name_scope('conv1'):
        conv1 = tf.nn.conv2d(x4d, weight_variable('conv1', [5, 5, 3, 64], 5e-2, 'w_conv1'), strides=[1, 1, 1, 1], padding='SAME')
        conv1 = tf.nn.bias_add(conv1, bias_variable('conv1', 0.0, tf.float32, [64], 'b_conv1'))
        conv1 = tf.nn.relu(conv1)
    # 池化层 1
    with tf.name_scope('pool1'):
        pool1 = tf.nn.max_pool(conv1, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding='SAME')
    # LRN层，Local Response Normalization
    with tf.name_scope('norm1'):
        norm1 = tf.nn.lrn(pool1, 4, bias=1.0, alpha=0.001/9.0, beta=0.75)
    # 卷积层 2
    with tf.name_scope('conv2'):
        conv2 = tf.nn.conv2d(norm1, weight_variable('conv2', [5, 5, 64, 64], 0.1, 'w_conv2'), strides=[1, 1, 1, 1], padding='SAME')
        conv2 = tf.nn.bias_add(conv2, bias_variable('conv2', 0.1, tf.float32, [64], 'b_conv2'))
        conv2 = tf.nn.relu(conv2)
    # LRN层，Local Response Normalization
    with tf.name_scope('norm2'):
        norm2 = tf.nn.lrn(conv2, 4, bias=1.0, alpha=0.001/9.0, beta=0.75)
    # 池化层 2
    with tf.name_scope('pool2'):
        pool2 = tf.nn.max_pool(norm2, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding='SAME')
    # 卷积层 3
#     conv3 = tf.nn.conv2d(conv2, w['conv3'], strides=[1, 1, 1, 1], padding='SAME')
#     conv3 = tf.nn.bias_add(conv3, b['conv3'])
#     conv3 = tf.nn.relu(conv3)
    # 池化层
#     pool3 = tf.nn.max_pool(conv3, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
    # 全连接层 1
    with tf.name_scope('reshape'):
        reshape = tf.reshape(pool2, [-1, 8*8*64])
    # dim = reshape.get_shape()[1].value
    with tf.name_scope('fc1'):
        fc1 = tf.add(tf.matmul(reshape, weight_variable('fc1', [8*8*64, n_fc1], 0.04, 'w_fc1')), bias_variable('fc1', 0.1, tf.float32, [n_fc1], 'b_fc1'))
        fc1 = tf.nn.relu(fc1)
    # 全连接层 2
    with tf.name_scope('fc2'):
        fc2 = tf.add(tf.matmul(fc1, weight_variable('fc2', [n_fc1, n_fc2], 0.1, 'w_fc2')), bias_variable('fc2', 0.1, tf.float32, [n_fc2], 'b_fc2'))
        fc2 = tf.nn.relu(fc2)
    # 全连接层 3, 即分类层
    with tf.name_scope('output'):
        fc3 = tf.add(tf.matmul(fc2, weight_variable('output', [n_fc2, n_classes], 1/192.0, 'w_output')), bias_variable('output', 0.0, tf.float32, [n_classes], 'b_outputs'))

    # 定义损失
    with tf.name_scope('cross_entropy'):
        cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=fc3, labels=y))
        tf.summary.scalar("cost_function", cost)
    with tf.name_scope('train'):
        optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)
    # 评估模型
    with tf.name_scope('accuracy'):
        with tf.name_scope('correct_prediction'):
            correct_pred = tf.equal(tf.argmax(fc3, 1), tf.argmax(y, 1))
        with tf.name_scope('accuracy'):
            accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))
    tf.summary.scalar('accuracy', accuracy)

    init = tf.global_variables_initializer()
    merged_summary_op = tf.summary.merge_all()
    saver = tf.train.Saver()

with tf.Session() as sess:
    sess.run(init)
    summary_writer = tf.summary.FileWriter('./tensorboard/log/', graph=tf.get_default_graph())
    # step = 1
    c = []
    total_batch = int(X_train.shape[0] / batch_size)
    for i in range(training_iters):
        avg_cost = 0
        for batch in range(total_batch):
            batch_x = X_train[batch*batch_size : (batch+1)*batch_size, :]
            batch_y = y_train[batch*batch_size : (batch+1)*batch_size, :]
            run_options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
            run_metadata = tf.RunMetadata()

#           batch_x, batch_y = nextbatch(X_train, y_train, batch_size=batch_size)
            _, co, summary = sess.run([optimizer, cost, merged_summary_op], feed_dict={x: batch_x, y: batch_y})
            summary_writer.add_run_metadata(run_metadata, 'step%d' % (i * total_batch + batch))
            summary_writer.add_summary(summary, i * total_batch + batch)
            saver.save(sess, './tensorboard/log/model.ckpt', i * total_batch + batch)
            avg_cost += co
            
        c.append(avg_cost)
        if (i+1) % display_step == 0:
            print("Iter " + str(i+1) + ", Training Loss= " + "{:.6f}".format(avg_cost))
    summary_writer.close()
    print("Optimization Finished!")

    # Test
    test_acc = sess.run(accuracy, feed_dict={x: X_test, y: y_test})
    print("Testing Accuracy:", test_acc)
    plt.plot(c)
    plt.xlabel('Iter')
    plt.ylabel('Cost')
    plt.title('lr=%f, ti=%d, bs=%d, acc=%f' % (learning_rate, training_iters, batch_size, test_acc))
    plt.tight_layout()
    plt.savefig('cnn-tf-cifar10-%s.png' % test_acc, dpi=200)
