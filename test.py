# -*- coding:utf-8 -*-
import numpy as np
import tensorflow as tf
from PIL import Image

img = Image.open('2.jpg')
img = img.convert('L')
img = np.array(img, dtype='float32')
img = img.astype(float)
img = np.reshape(img,[1, 784])

with tf.Session() as sess:
    new_saver = tf.train.import_meta_graph(r'.\model.ckpt.meta')
    new_saver.restore(sess, r'.\model.ckpt')
    graph = tf.get_default_graph()
    # sess.run(tf.global_variables_initializer())  这个不能有，有则计算图会初始化，导致读入的权重没有用了
    X = graph.get_tensor_by_name("input_x:0")
    keep_prob = graph.get_tensor_by_name('keep_prob:0')
    result = graph.get_tensor_by_name("prediction:0")
    #print(sess.run(result, feed_dict={X: img, keep_prob: 1.0}))
    out = tf.argmax(result, axis=1)
    print('预测结果:', sess.run(out, feed_dict={X: img, keep_prob: 1.0})[0])
