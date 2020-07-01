# -*- coding: utf-8 -*-
import numpy as np
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data


def new_weights(shape):
    return tf.Variable(tf.truncated_normal(shape, stddev=0.05))  # 产生截断正态分布随机数  stddev标准差


def new_biases(length):
    return tf.Variable(tf.constant(0.1, shape=length))


def conv2d(x, W):
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')


def max_pool_2x2(inputx):
    return tf.nn.max_pool(inputx, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')


# 读入MNIST数据集
data = input_data.read_data_sets('.', one_hot=True)
print("Size of:")
print("--Training-set:\t\t{}".format(len(data.train.labels)))
print("--Testing-set:\t\t{}".format(len(data.test.labels)))
print("--Validation-set:\t\t{}".format(len(data.validation.labels)))
data.test.cls = np.argmax(data.test.labels, axis=1)

# 网络模型搭建
x = tf.placeholder("float", shape=[None, 784], name="input_x")  # MNIST数据集图片尺寸为28*28*1
x_image = tf.reshape(x, [-1, 28, 28, 1])

y_true = tf.placeholder("float", shape=[None, 10], name='input_y')
y_true_cls = tf.argmax(y_true, dimension=1)

# Conv 1
layer_conv1 = {"weights": new_weights([5, 5, 1, 32]),  # kernel尺寸
               "biases": new_biases([32])}
h_conv1 = tf.nn.relu(conv2d(x_image, layer_conv1["weights"])+layer_conv1["biases"])
h_pool1 = max_pool_2x2(h_conv1)

# Conv 2
layer_conv2 = {"weights": new_weights([5, 5, 32, 64]),
               "biases": new_biases([64])}
h_conv2 = tf.nn.relu(conv2d(h_pool1, layer_conv2["weights"])+layer_conv2["biases"])
h_pool2 = max_pool_2x2(h_conv2)

# Full-connected layer 1
fc1_layer = {"weights": new_weights([7*7*64, 1024]),
             "biases": new_biases([1024])}
h_pool2_flat = tf.reshape(h_pool2, [-1, 7*7*64])
h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, fc1_layer["weights"])+fc1_layer["biases"])  # tf.matmul 矩阵相乘

# Dropout Layer
keep_prob = tf.placeholder("float", name="keep_prob")
h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)
# Full-connected layer 2
fc2_layer = {"weights": new_weights([1024, 10]),
             "biases": new_weights([10])}
# Predicted class
FC = tf.matmul(h_fc1_drop, fc2_layer["weights"])+fc2_layer["biases"]
y_pred = tf.nn.softmax(FC, name="prediction")  # The output is like [0 0 1 0 0 0 0 0 0 0]
y_pred_cls = tf.argmax(y_pred, dimension=1)  # Show the real predict number like '2'

# cost function to be optimized
cross_entropy = -tf.reduce_mean(y_true*tf.log(y_pred))  # 交叉熵函数
optimizer = tf.train.AdamOptimizer(learning_rate=1e-4).minimize(cross_entropy)
# Performance Measures
correct_prediction = tf.equal(y_pred_cls, y_true_cls)
accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))


saver = tf.train.Saver()  # max_to_keep=1
init = tf.global_variables_initializer()  # initialize_all_variables被代替了
sess = tf.Session()
# 上面定义的都没有运算，直到 sess.run 才会开始运算
sess.run(init)
for i in range(1, 2002):
    print(i)
    # training train_step 和 loss 都是由 placeholder 定义的运算，这里要用 feed 传入参数
    x_batch, y_true_batch = data.train.next_batch(36*i)
    feed_dict_train_op = {x: x_batch, y_true: y_true_batch, keep_prob: 0.5}  # keep_prob每个元素被保留的概率
    feed_dict_train = {x: x_batch, y_true: y_true_batch, keep_prob: 1.0}  # keep_prob:1就是所有元素全部保留的意思。
    sess.run(optimizer, feed_dict=feed_dict_train_op)
    if i % 50 == 0:
        # to see the step improvement
        acc = sess.run(accuracy, feed_dict=feed_dict_train)
        msg = "Optimization Iteration:{0:>6}, Training Accuracy: {1:>6.1%}"
        print(msg.format(i+1, acc))
    if i % 600 == 0:
        saver.save(sess=sess, save_path='/root/hand/model.ckpt')
