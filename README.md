# hand_gesture_recognition

<div align=center><img src="https://github.com/DQ0408/hand_gesture_recognition/blob/master/image/%E4%B8%BB%E7%A8%8B%E5%BA%8F.jpg"/></div>

<div align=center>Fig1.主界面</div>

<div align=center><img src="https://github.com/DQ0408/hand_gesture_recognition/blob/master/image/00499.jpg"/></div>

<div align=center>Fig2.图片模型训练结果</div>

### 验证集准确率为92%

## 识别流程：
OpenCV读取视频，用OpenPose提取手部的关键点，然后输入到全连接网络识别，识别结果可用于控制播放器

<div align=center><img src="https://github.com/DQ0408/hand_gesture_recognition/blob/master/image/openpose.jpg"/></div>

<div align=center>Fig3.OpenPose关键点图</div>

<div align=center><img src="https://github.com/DQ0408/hand_gesture_recognition/blob/master/image/%E4%B8%8A.jpg"/></div>

<div align=center>Fig4.上</div>

<div align=center><img src="https://github.com/DQ0408/hand_gesture_recognition/blob/master/image/%E4%B8%8B.jpg"/></div>

<div align=center>Fig5.下</div>

<div align=center><img src="https://github.com/DQ0408/hand_gesture_recognition/blob/master/image/%E5%B7%A6.jpg"/></div>

<div align=center>Fig6.左</div>

<div align=center><img src="https://github.com/DQ0408/hand_gesture_recognition/blob/master/image/%E5%8F%B3.jpg"/></div>

<div align=center>Fig7.右</div>

<div align=center><img src="https://github.com/DQ0408/hand_gesture_recognition/blob/master/image/%E5%BC%A0.jpg"/></div>

<div align=center>Fig7.张</div>

<div align=center><img src="https://github.com/DQ0408/hand_gesture_recognition/blob/master/image/%E5%90%88.jpg"/></div>

<div align=center>Fig8.合</div>

OpenPose手部模型下载：
[hand/pose_iter_102000.caffemodel](http://posefs1.perception.cs.cmu.edu/OpenPose/models/hand/pose_iter_102000.caffemodel)
[hand/pose_deploy.prototxt](https://github.com/CMU-Perceptual-Computing-Lab/openpose/blob/master/models/hand/pose_deploy.prototxt)



