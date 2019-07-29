# -- coding: utf-8 --
# author: ZQF  time: 2019/5/11 12:40


import sys
import cv2
from random import randint

trackerTypes = ['BOOSTING', 'MIL', 'GOTURN', 'TLD', 'MEDIANFLOW', 'KCF', 'MOSSE', 'CSRT']
trackerType = trackerTypes[-2]  # KCF, MOSSE is acceptable，KCF is the best，MOSSE is the fastest, CSRT is ok anyway


def adjust_frame(frame):  # do some rotate, clip and resize
    rows, cols, ch = frame.shape
    M = cv2.getRotationMatrix2D((cols, rows), 1, 1)
    frame = cv2.warpAffine(frame, M, (cols, rows))
    frame = frame[580:670, 470:1030]
    frame = cv2.resize(frame, None, fx=1.5, fy=1.5, interpolation=cv2.INTER_CUBIC)
    return frame


def createTrackerByName(trackerType):
    if trackerType == trackerTypes[0]:
        tracker = cv2.TrackerBoosting_create()
    elif trackerType == trackerTypes[1]:
        tracker = cv2.TrackerMIL_create()
    elif trackerType == trackerTypes[2]:
        tracker = cv2.TrackerGOTURN_create()
    elif trackerType == trackerTypes[3]:
        tracker = cv2.TrackerTLD_create()
    elif trackerType == trackerTypes[4]:
        tracker = cv2.TrackerMedianFlow_create()
    elif trackerType == trackerTypes[5]:
        tracker = cv2.TrackerKCF_create()
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

videoPath = r'.\4.MOV'  # the path of video
cap = cv2.VideoCapture(videoPath)  # open the video from the specified path

# read the first frame
ret, frame = cap.read()
frame = adjust_frame(frame)  # comment to cancel resize
if not ret:
    print('Failed to read video')
    sys.exit(1)

print('select an object and press Space or Enter to confirm, then the next object. ')
print('Press Esc to start tracking after all selection finished.')
bboxes = cv2.selectROIs("MultiTracker", frame, fromCenter=False, showCrosshair=True)

print('Selected bounding boxes {}'.format(bboxes))

# create lists to store trackers and colors
tracker_list = []
colors = []

for bbox in bboxes:
    tracker = createTrackerByName(trackerType)  # create a tracker
    ok = tracker.init(frame, tuple(bbox))  # initialize the tracking object in frame by bounding box
    if ok:
        tracker_list.append(tracker)
        colors.append((randint(64, 255), randint(64, 255), randint(64, 255)))

# 处理视频并跟踪对象
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    frame = adjust_frame(frame)  # comment to cancel resize

    timer = cv2.getTickCount()  # time point 1

    bboxes = []  # update new positions
    for tracker in tracker_list:
        ok, bbox = tracker.update(frame)
        if ok:  # if track successfully
            bboxes.append(bbox)
        else:  # else if track failed
            id = tracker_list.index(tracker)
            tracker_list.remove(tracker)
            colors.pop(id)

    # draw bounding box of tracking objects
    for i, newbox in enumerate(bboxes):
        p1 = (int(newbox[0]), int(newbox[1]))  # x, y
        p2 = (int(newbox[0] + newbox[2]), int(newbox[1] + newbox[3]))
        cv2.rectangle(frame, p1, p2, colors[i], 2, 1)

    fps = cv2.getTickFrequency() / (cv2.getTickCount() - timer)  # time point 2
    print(frame.shape)
    # cv2.putText(frame, "FPS : " + str(int(fps)), (10, 13), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (170, 50, 50), 2)
    # cv2.putText(frame, trackerType + " Tracker", (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (170, 50, 50), 2)
    cv2.putText(frame, "FPS : " + str(int(fps)), (720, 13), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (170, 50, 50), 2)
    cv2.putText(frame, trackerType + " Tracker", (720, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (170, 50, 50), 2)
    cv2.imshow('MultiTracker', frame)

    k = cv2.waitKey(1)
    if k == 27:  # press Ese to exit
        break
    elif k == ord('p'):  # press 'p' to draw a new object
        bbox = cv2.selectROI('MultiTracker', frame)  # return value: x, y, w, h
        tracker = createTrackerByName(trackerType)
        ok = tracker.init(frame, tuple(bbox))
        if ok:
            tracker_list.append(tracker)
            colors.append((randint(64, 255), randint(64, 255), randint(64, 255)))
