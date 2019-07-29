# -- coding: utf-8 --
# author: ZQF  time: 2019/3/19 20:40


import sys
import cv2
from random import randint

trackerTypes = ['BOOSTING', 'MIL', 'KCF', 'TLD', 'MEDIANFLOW', 'GOTURN', 'MOSSE', 'CSRT']
trackerType = trackerTypes[2]  # KCF, MOSSE的效果还能接受，KCF效果最好，MOSSE速度最快


def adjust_frame(frame):
    rows, cols, ch = frame.shape
    M = cv2.getRotationMatrix2D((cols, rows), 1, 1)  # 三个参数分别是旋转中心，旋转角度，比例
    frame = cv2.warpAffine(frame, M, (cols, rows))
    frame = frame[580:670, 470:1030]
    frame = cv2.resize(frame, None, fx=1.5, fy=1.5, interpolation=cv2.INTER_CUBIC)
    return frame


def createTrackerByName(trackerType):
    # 通过跟踪器的名字创建跟踪器
    if trackerType == trackerTypes[0]:
        tracker = cv2.TrackerBoosting_create()
    elif trackerType == trackerTypes[1]:
        tracker = cv2.TrackerMIL_create()
    elif trackerType == trackerTypes[2]:
        tracker = cv2.TrackerKCF_create()
    elif trackerType == trackerTypes[3]:
        tracker = cv2.TrackerTLD_create()
    elif trackerType == trackerTypes[4]:
        tracker = cv2.TrackerMedianFlow_create()
    elif trackerType == trackerTypes[5]:
        tracker = cv2.TrackerGOTURN_create()
    elif trackerType == trackerTypes[6]:
        tracker = cv2.TrackerMOSSE_create()
    elif trackerType == trackerTypes[7]:
        tracker = cv2.TrackerCSRT_create()
    else:
        tracker = None
        print('Incorrect tracker name')
        print('Available tracker name')

        for t in trackerTypes:
            print(t)
    return tracker


print('Default tracking algorithm is CSRT \n'
      'Available tracking algorithms are:\n')
for t in trackerTypes:
    print(t, end=' ')

videoPath = r'.\4.MOV'  # 设置加载的视频文件路径
cap = cv2.VideoCapture(videoPath)  # 创建video capture 来读取视频文件

# 读取第一帧
ret, frame = cap.read()
frame = adjust_frame(frame)
# 如果无法读取视频文件就退出
if not ret:
    print('Failed to read video')
    sys.exit(1)

# 选择框
bboxes = []
colors = []

# OpenCV的selectROI函数不适用于在Python中选择多个对象
# 所以循环调用此函数，直到完成选择所有对象
while True:
    # 在对象上绘制边界框selectROI的默认行为是从fromCenter设置为false时从中心开始绘制框，可以从左上角开始绘制框
    bbox = cv2.selectROI('MultiTracker', frame)  # 返回的四个值x, y, w, h
    bboxes.append(bbox)
    colors.append((randint(64, 255), randint(64, 255), randint(64, 255)))
    print("Press q to quit selecting boxes and start tracking")
    print("Press any other key to select next object")
    k = cv2.waitKey(0)
    if k == 113:  # q is pressed
        break

print('Selected bounding boxes {}'.format(bboxes))
# 初始化MultiTracker
# 有两种方法可以初始化multitracker
# 1. tracker = cv2.MultiTracker（“CSRT”）
# 所有跟踪器都添加到这个多路程序中
# 将使用CSRT算法作为默认值
# 2. tracker = cv2.MultiTracker（）
# 未指定默认算法
# 使用跟踪算法初始化MultiTracker
# 指定跟踪器类型

# 创建多跟踪器对象
multiTracker = cv2.MultiTracker_create()
# 初始化多跟踪器
for bbox in bboxes:
    multiTracker.add(createTrackerByName(trackerType), frame, bbox)

# 处理视频并跟踪对象
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    frame = adjust_frame(frame)

    timer = cv2.getTickCount()  # 计时点1
    # 获取后续帧中对象的更新位置
    ret, boxes = multiTracker.update(frame)  # 跟踪对象消失后boxes中的数据不会减少

    # 绘制跟踪的对象
    for i, newbox in enumerate(boxes):
        p1 = (int(newbox[0]), int(newbox[1]))  # x, y坐标
        p2 = (int(newbox[0] + newbox[2]), int(newbox[1] + newbox[3]))
        cv2.rectangle(frame, p1, p2, colors[i], 2, 1)
    fps = cv2.getTickFrequency() / (cv2.getTickCount() - timer)  # 计时点2
    cv2.putText(frame, "FPS : " + str(int(fps)), (10, 13), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (50, 170, 50), 2)
    cv2.putText(frame, trackerType + " Tracker", (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (50, 170, 50), 2)
    cv2.imshow('MultiTracker', frame)
    k = cv2.waitKey(1)
    if k == 27:
        break
    elif k == ord('p'):  # 按下p键可以新添加目标
        bbox = cv2.selectROI('MultiTracker', frame)  # 返回的四个值x, y, w, h
        bboxes.append(bbox)
        colors.append((randint(64, 255), randint(64, 255), randint(64, 255)))
        multiTracker.add(createTrackerByName(trackerType), frame, bbox)

