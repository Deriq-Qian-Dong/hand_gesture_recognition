# -*- coding: utf-8 -*-
"""
@author: Sucre
@email: qian.dong.2018@gmail.com
"""
import warnings

warnings.filterwarnings("ignore")
import os
import cv2
import time
import tkinter
import numpy as np
from tkinter import *
import tensorflow as tf


def fig_reg():
    '''
    实时识别图片接口
    '''
    labels = ["上", "下", "左", "右", "张", "合"]
    cap = cv2.VideoCapture(0)  # 开启本地摄像头
    # 设置每一帧的大小
    width = 640
    height = 480
    cap.set(3, width)
    cap.set(4, height)
    a = list(time.localtime(time.time()))
    a = [str(aa) for aa in a]
    out_path = "./data/recognition/" + "_".join(a)
    if not os.path.exists(out_path):
        os.makedirs(out_path)
    while True:
        ret, frame = cap.read()  # 从摄像头读取视频帧
        frameCopy = np.copy(frame)  # 复制一份用于画图
        frameWidth = frame.shape[1]
        frameHeight = frame.shape[0]
        aspect_ratio = frameWidth / frameHeight

        threshold = 0.2
        # 处理用于输入网络提取特征点
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

        cv2.imshow('Output-Keypoints', frameCopy)
        if cnt > 14:  # 如果符合条件，就进行识别
            # 读取预训练模型
            PATH = "./fig_model"
            sess = tf.Session(graph=tf.Graph())
            tf.saved_model.loader.load(sess, [tf.saved_model.tag_constants.TRAINING], PATH)
            output = sess.graph.get_tensor_by_name('predict:0')
            x = sess.graph.get_tensor_by_name('x:0')
            r = sess.run(output, feed_dict={x: np.array(points).reshape(-1, 44)})[0]
            sess.close()
            # 输出预测结果
            print(labels[r])
            cv2.imwrite('%s/fig.jpg' % out_path, frameCopy)
            print('结果保存在目录：', out_path)
            break
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()


def gen_feature_simplify(frames):
    # 提取视频特征
    start = time.time()
    threshold = 0.05  # 提取阈值
    max_point = 0  # 视频中最多的关键点
    max_fcnt = 15  # 视频中最多的关键点对应的帧数
    a = list(time.localtime(time.time()))
    a = [str(aa) for aa in a]
    out_path = "./data/recognition/" + "_".join(a)
    if not os.path.exists(out_path):
        os.makedirs(out_path)
    for i in range(15, 26, 2):  # 从第15帧开始提取
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
        feature = []
        if cnt >= max_point:  # 如果有更好的视频帧，则清空之前的特征
            try:
                feature = []
            except:
                pass
            max_point = cnt
            max_fcnt = f_cnt
            feature.append(points)
        if max_point >= 8:  # 如果视频帧中有大于等于8个关键点则跳出搜索
            break
    cv2.imwrite('%s/start.jpg' % out_path, frameCopy)
    f_cnt = max_fcnt + 10  # 提取最佳帧之后的第十帧作为动作结束帧
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
    feature.append(points)
    cv2.imwrite('%s/end.jpg' % out_path, frameCopy)
    print(time.time() - start)  # 输出提取特征时间
    print('结果保存在目录：', out_path)
    return np.array(feature).reshape(-1, 88)  # 返回特征


def video_reg_v2():
    # 识别定长视频
    cap = cv2.VideoCapture(0)
    width = 640
    height = 480
    cap.set(3, width)
    cap.set(4, height)
    frames = []
    cnt = 0
    while True:
        ret, frame = cap.read()
        frames.append(frame)  # 将视频帧保存到列表frames
        cv2.imshow('Output', frame)
        cnt += 1
        if cv2.waitKey(25) & 0xFF == ord('q') or cnt == 120:
            break
    cap.release()
    cv2.destroyAllWindows()
    ft = gen_feature_simplify(frames)  # 提取视频的特征
    #  识别视频的动作
    PATH = "./model"
    sess = tf.Session(graph=tf.Graph())
    tf.saved_model.loader.load(sess, [tf.saved_model.tag_constants.TRAINING], PATH)
    output = sess.graph.get_tensor_by_name('predict:0')
    x = sess.graph.get_tensor_by_name('x:0')
    r = sess.run(output, feed_dict={x: ft.reshape(-1, 88)})[0]
    sess.close()
    labels = ["上", "下", "左", "右", "张", "合"]
    print(labels[r])  # 输出预测结果


def video_reg():
    # 识别实时视频
    cap = cv2.VideoCapture(0)
    width = 640
    height = 480
    cap.set(3, width)
    cap.set(4, height)
    fps = cap.get(cv2.CAP_PROP_FPS)
    size = (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
            int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
    cnt = 0
    a = list(time.localtime(time.time()))
    a = [str(aa) for aa in a]
    out_path = "./data/recognition/" + "_".join(a)
    if not os.path.exists(out_path):
        os.makedirs(out_path)
    labels = ["上", "下", "左", "右", "张", "合"]
    while True:
        ret, frame = cap.read()
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

        cv2.imshow('Output-Keypoints', frameCopy)
        ft = []
        ft.append(points)
        if cnt > 14:
            # 如果满足条件，则开始提取动作结束帧的特征
            cv2.imwrite('%s/start.jpg' % out_path, frameCopy)
            ret, frame = cap.read()
            frameCopy = np.copy(frame)
            frameWidth = frame.shape[1]
            frameHeight = frame.shape[0]
            aspect_ratio = frameWidth / frameHeight

            threshold = 0.1

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
            cv2.imwrite('%s/end.jpg' % out_path, frameCopy)
            ft.append(points)
            ft = np.array(ft)
            # 读取预训练模型进行预测
            PATH = "./model"
            sess = tf.Session(graph=tf.Graph())
            tf.saved_model.loader.load(sess, [tf.saved_model.tag_constants.TRAINING], PATH)  # PATH还是路径
            output = sess.graph.get_tensor_by_name('predict:0')
            x = sess.graph.get_tensor_by_name('x:0')
            r = sess.run(output, feed_dict={x: ft.reshape(-1, 88)})[0]
            sess.close()
            print(labels[r])  # 打印预测结果
            print('结果保存在目录：', out_path)
            break
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    protoFile = "hand/pose_deploy.prototxt"
    weightsFile = "hand/pose_iter_102000.caffemodel"
    nPoints = 22
    POSE_PAIRS = [[0, 1], [1, 2], [2, 3], [3, 4], [0, 5], [5, 6], [6, 7], [7, 8],
                  [0, 9], [9, 10], [10, 11], [11, 12],
                  [0, 13], [13, 14], [14, 15], [15, 16], [0, 17], [17, 18], [18, 19], [19, 20]]
    net = cv2.dnn.readNetFromCaffe(protoFile, weightsFile)
    root = tkinter.Tk(className="识别程序")
    root.geometry('800x500+800+500')
    fig_reg = Button(root, text="识别图片", command=fig_reg)
    fig_reg.pack()
    video_reg = Button(root, text="识别实时视频", command=video_reg)
    video_reg.pack()
    video_reg_2 = Button(root, text="识别定长视频", command=video_reg_v2)
    video_reg_2.pack()
    root.mainloop()
