# -*- coding: utf-8 -*-
"""
Created on 2019/01/23 14:41
@author: Sucre
@email: qian.dong.2018@gmail.com
"""
import warnings

warnings.filterwarnings("ignore")
import os
import cv2
import tkinter
import numpy as np
from tkinter import *
from tkinter import ttk


def fig_collector():
    top = Toplevel()
    top.geometry("400x200")
    top.title('请输入英文的姓名')

    v1 = StringVar()
    v1.set('输入拼音姓名')
    e1 = Entry(top, textvariable=v1, width=10)
    e1.grid(row=1, column=0, padx=1, pady=1)
    Button(top, text='确定', command=top.quit).grid(row=1, column=1, padx=1, pady=1)
    top.mainloop()
    p_name = e1.get()
    top.destroy()

    top = Toplevel()
    top.geometry("400x200")
    top.title('请选择动作的标签')

    comvalue = tkinter.StringVar()  # 窗体自带的文本，新建一个值
    comboxlist = ttk.Combobox(top, textvariable=comvalue)  # 初始化
    labels = ["上", "下", "左", "右", "张", "合"]
    comboxlist["values"] = tuple(labels)
    comboxlist.current(0)  # 选择第一个
    comboxlist.pack()
    btn = Button(top, text='确定', command=top.quit)
    btn.pack()
    top.mainloop()
    l_name = comboxlist.get()
    top.destroy()
    out_dir = "./data/figure/%s/%d" % (p_name, labels.index(l_name))
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
    cap = cv2.VideoCapture(0)
    width = 640
    height = 480
    cap.set(3, width)
    cap.set(4, height)
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
        if cnt > 14:
            print('done')
            cv2.imwrite('%s/%s.jpg' % (out_dir, str(labels.index(l_name))), frame)
            cv2.imwrite('%s/%s_copy.jpg' % (out_dir, str(labels.index(l_name))), frameCopy)
            np.save('%s/%s.npy' % (out_dir, str(labels.index(l_name))), points)
            break
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()


def video_collector():
    top = Toplevel()
    top.geometry("400x200")
    top.title('请输入英文的姓名')

    v1 = StringVar()
    v1.set('输入拼音姓名')
    e1 = Entry(top, textvariable=v1, width=10)
    e1.grid(row=1, column=0, padx=1, pady=1)
    Button(top, text='确定', command=top.quit).grid(row=1, column=1, padx=1, pady=1)
    top.mainloop()
    p_name = e1.get()
    top.destroy()

    top = Toplevel()
    top.geometry("400x200")
    top.title('请选择动作的标签')

    comvalue = tkinter.StringVar()  # 窗体自带的文本，新建一个值
    comboxlist = ttk.Combobox(top, textvariable=comvalue)  # 初始化
    labels = ["上", "下", "左", "右", "张", "合"]
    comboxlist["values"] = tuple(labels)
    comboxlist.current(0)  # 选择第一个
    comboxlist.pack()
    btn = Button(top, text='确定', command=top.quit)
    btn.pack()
    top.mainloop()
    l_name = comboxlist.get()
    top.destroy()
    out_dir = "./data/video/%s/%d" % (p_name, labels.index(l_name))
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    cap = cv2.VideoCapture(0)
    width = 640
    height = 480
    cap.set(3, width)
    cap.set(4, height)
    fps = cap.get(cv2.CAP_PROP_FPS)
    print(cv2.CAP_PROP_FPS)
    size = (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
            int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
    out = cv2.VideoWriter(out_dir + '/%d.avi' % labels.index(l_name), cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'), fps,
                          size)
    cnt = 0
    while True:
        ret, frame = cap.read()
        out.write(frame)
        cv2.imshow('Output', frame)
        cnt += 1
        if cv2.waitKey(25) & 0xFF == ord('q') or cnt == 120:
            print('done')
            break
    cap.release()
    out.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    protoFile = "hand/pose_deploy.prototxt"
    weightsFile = "hand/pose_iter_102000.caffemodel"
    nPoints = 22
    POSE_PAIRS = [[0, 1], [1, 2], [2, 3], [3, 4], [0, 5], [5, 6], [6, 7], [7, 8],
                  [0, 9], [9, 10], [10, 11], [11, 12],
                  [0, 13], [13, 14], [14, 15], [15, 16], [0, 17], [17, 18], [18, 19], [19, 20]]
    net = cv2.dnn.readNetFromCaffe(protoFile, weightsFile)
    root = tkinter.Tk(className="采集程序")
    root.geometry('800x500+800+500')
    fig_col = Button(root, text="采集图片", command=fig_collector)
    fig_col.pack()
    video_col = Button(root, text="采集视频", command=video_collector)
    video_col.pack()
    root.mainloop()
