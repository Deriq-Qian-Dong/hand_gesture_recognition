# -*- coding: utf-8 -*-
"""
@author: Sucre
@email: qian.dong.2018@gmail.com
"""
# 图片的模型
import os
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn import preprocessing
from sklearn.model_selection import train_test_split


def get_data():
    '''
    获取图片特征数据
    '''
    data = []
    target = []
    for root, dirs, files in os.walk("./data/figure"):  # 游走图片数据文件夹，获取特征文件
        for name in files:
            if "npy" in name:
                npy = os.path.join(root, name)
                data.append(np.load(npy))
                target.append(npy.split("\\")[-2])
    data = np.array(data).reshape(-1, 44)  # 将特征转化为44维的向量
    target = np.array(target).astype(int).reshape(-1, 1)
    oe = preprocessing.OneHotEncoder()
    oe.fit(target)
    target = np.array(oe.transform(target).todense())  # 将特征独热编码
    return data, target


def train_model():
    x_data, y_data = get_data()
    X_train, X_test, y_train, y_test = train_test_split(x_data, y_data, test_size=0.2)  # 切分数据集

    def add_layer(inputs, in_size, out_size, activation_function=None):  # 添加网络层
        Weights = tf.Variable(tf.random_normal([in_size, out_size], stddev=0.01))
        biases = tf.Variable(tf.zeros([1, out_size]) + 0.1)
        Wx_plus_b = tf.matmul(inputs, Weights) + biases
        if activation_function is None:
            outputs = Wx_plus_b
        else:
            outputs = activation_function(Wx_plus_b)
        return outputs
    # 定义网络图结构
    x = tf.placeholder(tf.float32, shape=[None, 44], name="x")
    y = tf.placeholder(tf.float32, shape=[None, 6], name="y")
    l1 = add_layer(x, 44, 10, activation_function=tf.nn.relu)
    prediction = add_layer(l1, 10, 6, activation_function=None)
    prediction = tf.nn.softmax(prediction)
    predict = tf.argmax(prediction, 1, name="predict")  # 定义变量，用于获得预测结果
    cross_entropy = tf.reduce_mean(-tf.reduce_sum(y * tf.log(prediction), reduction_indices=[1]))
    correct_prediction = tf.equal(tf.argmax(prediction, 1), tf.argmax(y, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)
    init = tf.global_variables_initializer()
    sess = tf.Session()
    sess.run(init)
    accs = []
    losses = []
    val_accs = []
    val_losses = []
    epochs = []
    f, axs = plt.subplots(4, 1, figsize=(10, 5))  # 在一张图上画四个子图
    for epoch in range(500):
        sess.run(train_step, feed_dict={x: X_train, y: y_train})
        acc = sess.run(accuracy, feed_dict={x: X_train, y: y_train})
        loss = sess.run(cross_entropy, feed_dict={x: X_train, y: y_train})
        val_acc = sess.run(accuracy, feed_dict={x: X_test, y: y_test})
        val_loss = sess.run(cross_entropy, feed_dict={x: X_test, y: y_test})
        accs.append(acc)
        losses.append(loss)
        val_accs.append(val_acc)
        val_losses.append(val_loss)
        epochs.append(epoch)
        print("epoch:%d" % epoch, "training loss %.4f" % loss, "training acc %.4f" % acc,
              "eval loss %.4f" % val_loss, "eval acc %.4f" % val_acc)
        # 画图并保存每一步的结果
        axs[0].plot(epochs, losses, c='b', marker='.', label='training loss')
        axs[1].plot(epochs, accs, c='r', marker='.', label='training acc')
        axs[2].plot(epochs, val_losses, c='b', marker='.', label='eval loss')
        axs[3].plot(epochs, val_accs, c='r', marker='.', label='eval acc')
        if epoch == 0:
            for i in range(4):
                axs[i].legend(loc='best')
        plt.pause(0.000001)
        if not os.path.exists('./figs'):
            os.makedirs('./figs')
        plt.savefig('./figs/%s.jpg' % str(epoch).zfill(5))
    # 保存模型用于预测
    PATH = "./fig_model"
    builder = tf.saved_model.builder.SavedModelBuilder(PATH)
    builder.add_meta_graph_and_variables(sess, [tf.saved_model.tag_constants.TRAINING])
    builder.save()


if __name__ == '__main__':
    train_model()
