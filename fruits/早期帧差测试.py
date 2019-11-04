# -- coding:utf-8 --
# author: ZQF time:2018/12/15


import cv2
import numpy as np


def nothing(x):
    pass


def get_wh(box):
    dx1 = box[0][1] - box[1][1]
    dy1 = box[0][0] - box[1][0]
    dx2 = box[1][1] - box[2][1]
    dy2 = box[1][0] - box[2][0]
    w = (dx1**2 + dy1**2)**0.5
    h = (dx2**2 + dy2**2)**0.5
    if w > h:
        bi = w / h
    else:
        bi = h / w
    print('\t长宽比为：', bi)


def do_morph(diff):  # 形态学处理
    x = cv2.getTrackbarPos('x', 'pic0')
    y = cv2.getTrackbarPos('y', 'pic0')
    if x < 1:
        x = 1
    if y < 1:
        y = 1
    ksize = (x, y)
    morph = [cv2.MORPH_RECT, cv2.MORPH_ELLIPSE, cv2.MORPH_CROSS]
    i = cv2.getTrackbarPos('morph', 'pic0')
    kernel = cv2.getStructuringElement(morph[i], ksize)
    diff = cv2.dilate(diff, kernel, iterations=2)
    diff = cv2.morphologyEx(diff, cv2.MORPH_CLOSE, kernel, iterations=2)
    diff  = cv2.erode(diff, kernel)
    return diff


def print_characters(cnt):
    M = cv2.moments(cnt)
    huM = cv2.HuMoments(M)
    if M['m00'] >= 6:
        cx = int(M['m10'] / M['m00'])
        cy = int(M['m01'] / M['m00'])
        print('\t轮廓的重心是:', cx, cy)
        print("\t轮廓的面积是:", cv2.contourArea(cnt))
        print('\t轮廓的周长是:', cv2.arcLength(cnt, True))
        # (x, y), (MA, ma), angle = cv2.fitEllipse(cnt)
        # print('angle:', angle)
        print('\tM:', M)
        print('\thuM:', list(huM))


cv2.namedWindow('pic0')
cv2.createTrackbar('x', 'pic0', 1, 20, nothing)
cv2.createTrackbar('y', 'pic0', 1, 20, nothing)
cv2.createTrackbar('morph', 'pic0', 0, 2, nothing)
cv2.createTrackbar('num', 'pic0', 1, 6, nothing)


def draw_rect(cnts):
    j = 0
    for i, cnt in enumerate(cnts):
        area = cv2.contourArea(cnt)
        if area < 200:
            continue
        # cv2.drawContours(pic0, cnts, i, (0, 0, 255), 2)
        rect = cv2.minAreaRect(cnt)  # 找到包围输入2D点集的最小区域的旋转矩形
        box = cv2.boxPoints(rect)  # 获得旋转矩形的四个顶点
        box = np.int0(box)
        j += 1
        print('第%d辆车：' % j)
        cv2.drawContours(pic0, [box], 0, (255, 0, 255), 2)
        cv2.putText(pic0, str(j), tuple(box[0]), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2, cv2.LINE_AA)
        print_characters(box)  # 输出画出矩形轮廓的参数
        get_wh(box)
    print('\n--------------------------------------------------------------------------------------')


def open_pic():
    global pic0, diff
    num = cv2.getTrackbarPos('num', 'pic0')
    pic1 = cv2.imread('car_pic%d.jpg' % num)
    pic2 = cv2.imread('car_pic%d.jpg' % (num + 1))
    pic0 = pic1.copy()
    pic1 = cv2.cvtColor(pic1, cv2.COLOR_BGR2GRAY)
    pic2 = cv2.cvtColor(pic2, cv2.COLOR_BGR2GRAY)
    diff = cv2.absdiff(pic1, pic2)
    diff = cv2.GaussianBlur(diff, (5, 5), 0)
    _, diff = cv2.threshold(diff, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    # diff = cv2.adaptiveThreshold(diff, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, 3, 2)
    diff = cv2.GaussianBlur(diff, (5, 5), 0)


while 1:
    open_pic()  # 打开图片

    diff = do_morph(diff)  # 做形态学变换

    cv2.imshow('diff', diff)

    _, cnts, heri = cv2.findContours(diff, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    draw_rect(cnts)  # 绘制轮廓边界矩形

    cv2.imshow('pic0', pic0)

    k = cv2.waitKey(0)  # 按一下键盘刷新一次
    if k == 27:
        break

cv2.destroyAllWindows()
