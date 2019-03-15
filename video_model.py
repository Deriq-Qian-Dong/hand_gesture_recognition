# -*- coding: utf-8 -*-
"""
@author: Sucre
@email: qian.dong.2018@gmail.com
"""
import os
import cv2
import time
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn import preprocessing
from sklearn.model_selection import train_test_split


def get_vids():
    '''
    获取视频的路径
    '''
    vids = []
    target = []
    for root, dirs, files in os.walk("./data"):
        for name in files:
            if "avi" in name:
                vids.append(os.path.join(root, name))
                target.append(name.split(".avi")[0])
    return vids, target


def gen_feature():
    '''
    提取视频特征，，耗时太久，弃用
    '''
    vids, target = get_vids()
    protoFile = "hand/pose_deploy.prototxt"
    weightsFile = "hand/pose_iter_102000.caffemodel"
    nPoints = 22
    POSE_PAIRS = [[0, 1], [1, 2], [2, 3], [3, 4], [0, 5], [5, 6], [6, 7], [7, 8],
                  [0, 9], [9, 10], [10, 11], [11, 12],
                  [0, 13], [13, 14], [14, 15], [15, 16], [0, 17], [17, 18], [18, 19], [19, 20]]
    net = cv2.dnn.readNetFromCaffe(protoFile, weightsFile)
    for vid in vids:
        start = time.time()
        cap = cv2.VideoCapture(vid)

        frames = []
        while True:
            ret, frame = cap.read()
            if frame is None:
                break
            frames.append(frame)
        cap.release()
        cv2.destroyAllWindows()
        print(vid, len(frames))
        for f_cnt, frame in enumerate(frames):
            frameCopy = np.copy(frame)
            frameWidth = frame.shape[1]
            frameHeight = frame.shape[0]
            aspect_ratio = frameWidth / frameHeight

            threshold = 0.2

            # input image dimensions for the network
            inHeight = 368
            inWidth = int(((aspect_ratio * inHeight) * 8) // 8)
            inpBlob = cv2.dnn.blobFromImage(frame, 1.0 / 255, (inWidth, inHeight), (0, 0, 0), swapRB=False, crop=False)

            net.setInput(inpBlob)

            output = net.forward()
            # Empty list to store the detected keypoints
            points = []
            cnt = 0
            for i in range(nPoints):
                # confidence map of corresponding body's part.
                probMap = output[0, i, :, :]
                probMap = cv2.resize(probMap, (frameWidth, frameHeight))

                # Find global maxima of the probMap.
                minVal, prob, minLoc, point = cv2.minMaxLoc(probMap)

                if prob > threshold:
                    cv2.circle(frameCopy, (int(point[0]), int(point[1])), 8, (0, 255, 255), thickness=-1,
                               lineType=cv2.FILLED)

                    # Add the point to the list if the probability is greater than the threshold
                    points.append((int(point[0]), int(point[1])))
                    cnt += 1
                else:
                    points.append((0, 0))
            if cnt > 14:
                print(vid, len(frames), f_cnt, 'done')
                cv2.imwrite(vid.replace(".avi", "_%d.jpg" % f_cnt), frame)
                cv2.imwrite(vid.replace(".avi", "_%d_copy.jpg" % f_cnt), frameCopy)
                np.save(vid.replace(".avi", "_%d.npy" % f_cnt), points)
                break
        for f_cnt, frame in enumerate(frames[::-1]):
            print(f_cnt)
            f_cnt = len(frames) - 1 - f_cnt
            print(f_cnt)
            frameCopy = np.copy(frame)
            frameWidth = frame.shape[1]
            frameHeight = frame.shape[0]
            aspect_ratio = frameWidth / frameHeight

            threshold = 0.2

            # input image dimensions for the network
            inHeight = 368
            inWidth = int(((aspect_ratio * inHeight) * 8) // 8)
            inpBlob = cv2.dnn.blobFromImage(frame, 1.0 / 255, (inWidth, inHeight), (0, 0, 0), swapRB=False, crop=False)

            net.setInput(inpBlob)

            output = net.forward()
            # Empty list to store the detected keypoints
            points = []
            cnt = 0
            for i in range(nPoints):
                # confidence map of corresponding body's part.
                probMap = output[0, i, :, :]
                probMap = cv2.resize(probMap, (frameWidth, frameHeight))

                # Find global maxima of the probMap.
                minVal, prob, minLoc, point = cv2.minMaxLoc(probMap)

                if prob > threshold:
                    cv2.circle(frameCopy, (int(point[0]), int(point[1])), 8, (0, 255, 255), thickness=-1,
                               lineType=cv2.FILLED)

                    # Add the point to the list if the probability is greater than the threshold
                    points.append((int(point[0]), int(point[1])))
                    cnt += 1
                else:
                    points.append((0, 0))
            if cnt > 14:
                print(vid, len(frames), f_cnt, 'done')
                cv2.imwrite(vid.replace(".avi", "_%d.jpg" % f_cnt), frame)
                cv2.imwrite(vid.replace(".avi", "_%d_copy.jpg" % f_cnt), frameCopy)
                np.save(vid.replace(".avi", "_%d.npy" % f_cnt), points)
                break
        print(time.time() - start)


def gen_feature_simplify():
    '''
    提取视频特征
    '''
    vids, target = get_vids()
    protoFile = "hand/pose_deploy.prototxt"
    weightsFile = "hand/pose_iter_102000.caffemodel"
    nPoints = 22
    net = cv2.dnn.readNetFromCaffe(protoFile, weightsFile)
    threshold = 0.05

    for vid in vids:
        start = time.time()
        #  将视频读入frames
        cap = cv2.VideoCapture(vid)
        frames = []
        while True:
            ret, frame = cap.read()
            if frame is None:
                break
            frames.append(frame)
        cap.release()
        cv2.destroyAllWindows()
        print(vid, len(frames))
        max_point = 0
        max_fcnt = 15
        for i in range(15, 26, 2):  # 开始提取特征，从第15帧数开始搜索动作开始最佳帧
            f_cnt = i
            frame = frames[f_cnt]
            frameCopy = np.copy(frame)
            frameWidth = frame.shape[1]
            frameHeight = frame.shape[0]
            aspect_ratio = frameWidth / frameHeight

            # input image dimensions for the network
            inHeight = 368
            inWidth = int(((aspect_ratio * inHeight) * 8) // 8)
            inpBlob = cv2.dnn.blobFromImage(frame, 1.0 / 255, (inWidth, inHeight), (0, 0, 0), swapRB=False, crop=False)

            net.setInput(inpBlob)

            output = net.forward()
            # Empty list to store the detected keypoints
            points = []
            cnt = 0
            for i in range(nPoints):
                # confidence map of corresponding body's part.
                probMap = output[0, i, :, :]
                probMap = cv2.resize(probMap, (frameWidth, frameHeight))

                # Find global maxima of the probMap.
                minVal, prob, minLoc, point = cv2.minMaxLoc(probMap)

                if prob > threshold:
                    cv2.circle(frameCopy, (int(point[0]), int(point[1])), 8, (0, 255, 255), thickness=-1,
                               lineType=cv2.FILLED)

                    # Add the point to the list if the probability is greater than the threshold
                    points.append((int(point[0]), int(point[1])))
                    cnt += 1
                else:
                    points.append((0, 0))
            print(f_cnt, cnt)
            if cnt > max_point:
                try:
                    os.remove(vid.replace(".avi", "_%d.jpg" % max_fcnt))
                    os.remove(vid.replace(".avi", "_%d_copy.jpg" % max_fcnt))
                    os.remove(vid.replace(".avi", "_%d.npy" % max_fcnt))
                except:
                    pass
                max_point = cnt
                max_fcnt = f_cnt
                print(vid, len(frames), f_cnt, 'done')
                cv2.imwrite(vid.replace(".avi", "_%d.jpg" % max_fcnt), frame)
                cv2.imwrite(vid.replace(".avi", "_%d_copy.jpg" % max_fcnt), frameCopy)
                np.save(vid.replace(".avi", "_%d.npy" % max_fcnt), points)
            if max_point >= 8:
                break
        #  动作开始最佳帧后第十帧作为动作结束帧
        f_cnt = max_fcnt + 10
        frame = frames[f_cnt]
        frameCopy = np.copy(frame)
        frameWidth = frame.shape[1]
        frameHeight = frame.shape[0]
        aspect_ratio = frameWidth / frameHeight

        # input image dimensions for the network
        inHeight = 368
        inWidth = int(((aspect_ratio * inHeight) * 8) // 8)
        inpBlob = cv2.dnn.blobFromImage(frame, 1.0 / 255, (inWidth, inHeight), (0, 0, 0), swapRB=False, crop=False)

        net.setInput(inpBlob)

        output = net.forward()
        # Empty list to store the detected keypoints
        points = []
        cnt = 0
        for i in range(nPoints):
            # confidence map of corresponding body's part.
            probMap = output[0, i, :, :]
            probMap = cv2.resize(probMap, (frameWidth, frameHeight))

            # Find global maxima of the probMap.
            minVal, prob, minLoc, point = cv2.minMaxLoc(probMap)

            if prob > threshold:
                cv2.circle(frameCopy, (int(point[0]), int(point[1])), 8, (0, 255, 255), thickness=-1,
                           lineType=cv2.FILLED)

                # Add the point to the list if the probability is greater than the threshold
                points.append((int(point[0]), int(point[1])))
                cnt += 1
            else:
                points.append((0, 0))
        print(vid, len(frames), f_cnt, 'done')
        cv2.imwrite(vid.replace(".avi", "_%d.jpg" % f_cnt), frame)
        cv2.imwrite(vid.replace(".avi", "_%d_copy.jpg" % f_cnt), frameCopy)
        np.save(vid.replace(".avi", "_%d.npy" % f_cnt), points)
        print(time.time() - start)


def clean_history():
    # 清理提取的特征，测试时用的
    for root, dirs, files in os.walk("./data/video"):
        for name in files:
            if "npy" in name or "jpg" in name:
                os.remove(os.path.join(root, name))
                print(os.path.join(root, name))


def copy_history():
    import shutil
    # 复制提取的特征，测试时用的
    for root, dirs, files in os.walk("./data/video"):
        for name in files:
            if "npy" in name or "jpg" in name:
                target = os.path.join(root, name).replace('video', 'feature')
                if not os.path.exists(os.path.dirname(target)):
                    os.makedirs(os.path.dirname(target))
                shutil.copy(os.path.join(root, name), target)
                print(os.path.join(root, name))


def get_data():
    '''
    获取视频的特征数据
    '''
    data = []
    target = []
    nums = ['0', '1', '2', '3', '4', '5']
    for root, dirs, files in os.walk("./data/feature"):
        for name in dirs:
            if name in nums:
                if len(os.listdir(os.path.join(root, name))) == 6:
                    one_data = []
                    for npy in os.listdir(os.path.join(root, name)):
                        if "npy" in npy:
                            one_data.append(np.load(os.path.join(root, name, npy)))
                    data.append(one_data)
                    target.append(name)
    data = np.array(data).reshape(-1, 2*44)
    target = np.array(target).astype(int).reshape(-1, 1)
    oe = preprocessing.OneHotEncoder()
    oe.fit(target)
    target = np.array(oe.transform(target).todense())
    return data, target


def prelu(_x):
    '''
    带倾斜的激活函数
    '''
    alphas = tf.get_variable('alpha', _x.get_shape()[-1],
                             initializer=tf.constant_initializer(0.0),
                             dtype=tf.float32)
    pos = tf.nn.relu(_x)
    neg = alphas * (_x - abs(_x)) * 0.5
    return pos + neg


class Model:
    def __init__(self):
        self.class_num = 6
        self.lr = 1e-6
        self.keep = 1
        self.x = tf.placeholder(tf.float32, shape=[None, 88], name="x")
        self.y = tf.placeholder(tf.float32, shape=[None, 6], name="y")
        # 网络结构图  三个全连接的残差全连接网络  用adam优化交叉熵
        self.result = tf.nn.dropout(self.add_basic_block(self.x, in_size=88, hidden=512, out=256, name="1"), 1)
        self.result = tf.nn.dropout(self.add_basic_block(self.result, in_size=256, hidden=64, out=128, name="2"), 1)
        self.result = tf.nn.dropout(self.add_basic_block(self.result, in_size=128, hidden=32, out=6, name="3"), 1)
        # self.result = tf.nn.dropout(self.add_basic_block(self.result, in_size=32, hidden=8, out=6, name="4"), 1)

        self.output = tf.nn.softmax(self.result, name="output")
        self.predict = tf.argmax(self.output, 1, name="predict")
        self.loss = tf.reduce_mean(-tf.reduce_sum(self.y * tf.log(self.output), reduction_indices=[1]))
        self.train_step = tf.train.AdamOptimizer(self.lr).minimize(self.loss)
        corrects = tf.equal(tf.argmax(self.output, 1), tf.argmax(self.y, 1))
        self.acc = tf.reduce_mean(tf.cast(corrects, tf.float32))
        self.init = tf.global_variables_initializer()
        self.sess = tf.Session()
        writer = tf.summary.FileWriter('./logs/', self.sess.graph)
        self.sess.run(self.init)

    def add_layer(self, inputs, in_size, out_size, activation_function=None):
        Weights = tf.Variable(tf.random_normal([in_size, out_size], stddev=0.01))
        biases = tf.Variable(tf.zeros([1, out_size]) + 0.1)
        Wx_plus_b = tf.matmul(inputs, Weights) + biases
        if activation_function is None:
            outputs = Wx_plus_b
        else:
            outputs = activation_function(Wx_plus_b)
        return outputs

    def add_basic_block(self, x, in_size, hidden, out, name):
        output1_1 = self.add_layer(inputs=x, in_size=in_size, out_size=out, activation_function=None)
        with tf.variable_scope('%s_output1_1' % name):
            output1_1 = prelu(output1_1)
        result = output1_1
        output1_2 = self.add_layer(inputs=output1_1, in_size=out, out_size=hidden, activation_function=None)
        with tf.variable_scope('%s_output1_2' % name):
            output1_2 = prelu(output1_2)
        output1_3 = self.add_layer(inputs=output1_2, in_size=hidden, out_size=out, activation_function=None)
        with tf.variable_scope('%s_output1_3' % name):
            output1_3 = prelu(output1_3)
        result += output1_3
        return result


def train_model():
    # 类似图片的train_model
    x_data, y_data = get_data()
    X_train, X_test, y_train, y_test = train_test_split(x_data, y_data, test_size=0.2, random_state=20)  # 20--78
    res_model = Model()
    accs = []
    losses = []
    val_accs = []
    val_losses = []
    epochs = []
    f, axs = plt.subplots(4, 1, figsize=(10, 5))
    for epoch in range(5000):
        res_model.sess.run(res_model.train_step, feed_dict={res_model.x: X_train, res_model.y: y_train})
        acc = res_model.sess.run(res_model.acc, feed_dict={res_model.x: X_train, res_model.y: y_train})
        loss = res_model.sess.run(res_model.loss, feed_dict={res_model.x: X_train, res_model.y: y_train})
        val_acc = res_model.sess.run(res_model.acc, feed_dict={res_model.x: X_test, res_model.y: y_test})
        val_loss = res_model.sess.run(res_model.loss, feed_dict={res_model.x: X_test, res_model.y: y_test})
        accs.append(acc)
        losses.append(loss)
        val_accs.append(val_acc)
        val_losses.append(val_loss)
        epochs.append(epoch)
        print("epoch:%d" % epoch, "training loss %.4f" % loss, "training acc %.4f" % acc,
              "eval loss %.4f" % val_loss, "eval acc %.4f" % val_acc, "lr:", res_model.lr)
        if epoch == 0:
            axs[0].plot(epochs, losses, c='b', marker='.', label='training loss')
            axs[1].plot(epochs, accs, c='r', marker='.', label='training acc')
            axs[2].plot(epochs, val_losses, c='b', marker='.', label='eval loss')
            axs[3].plot(epochs, val_accs, c='r', marker='.', label='eval acc')
            if epoch == 0:
                for i in range(4):
                    axs[i].legend(loc='best')
            plt.pause(0.000001)
            if not os.path.exists('./v_figs'):
                os.makedirs('./v_figs')
            plt.savefig('./v_figs/%s.jpg' % str(epoch).zfill(5))
    axs[0].plot(epochs, losses, c='b', marker='.', label='training loss')
    axs[1].plot(epochs, accs, c='r', marker='.', label='training acc')
    axs[2].plot(epochs, val_losses, c='b', marker='.', label='eval loss')
    axs[3].plot(epochs, val_accs, c='r', marker='.', label='eval acc')
    plt.savefig('./v_figs/%s.jpg' % str(epoch).zfill(5))
    PATH = "./model"
    builder = tf.saved_model.builder.SavedModelBuilder(PATH)
    builder.add_meta_graph_and_variables(res_model.sess, [tf.saved_model.tag_constants.TRAINING])
    builder.save()



if __name__ == '__main__':
    # clean_history()
    # gen_feature_simplify()
    train_model()
