import math
import numpy as np
import cv2 as cv
from keras.models import load_model
from Common import *


def levelAngle(x1, y1, x2, y2):  # 求角度 指针向量的两个点
    """
    :param x1: point1_x
    :param y1: point1_y
    :param x2: point2_x
    :param y2: point2_x
    :return: angle
    """
    if (y1 < y2):
        vector = (x1 - x2, y1 - y2)
    else:
        vector = (x2 - x1, y2 - y1)
    cos = (vector[0] * (200.0)) / (200.0 * math.sqrt(vector[0] ** 2 + vector[1] ** 2))
    return 180 - math.acos(cos) * 180.0 / math.pi


def line_detect(gray, low_thres=40, high_thres=110):
    """
    :param gray: graph
    :param low_thres: Canny边缘检测的低阈值
    :param high_thres: Canny边缘检测的高阈值
    :return: two points of line
    """
    edges = cv.Canny(gray, low_thres, high_thres, apertureSize=3)
    linepoint = []
    lines = cv.HoughLinesP(edges, 1, np.pi / 180, 30, minLineLength=25, maxLineGap=2)
    try:
        for line in lines:
            linepoint.append(line[0])
    except:
        linepoint.append((0, 0, 1, 0))  # 随机给一条直线
    return linepoint


def bileiqi1(src, info):
    """
    :param src: ROI
    :return:value
    """
    src = meterFinderBySIFT(src, info["template"])
    # cv2.imshow("src", src)
    # cv2.waitKey(0)

    low_thres = 40  # Canny检测的高低阈值
    high_thres = 110

    src_gray = cv.cvtColor(src, cv.COLOR_BGR2GRAY)
    src_gray = cv.equalizeHist(src_gray)
    thres, src_bin = cv.threshold(src_gray, 100, 255, cv.THRESH_BINARY)
    kernel = cv.getStructuringElement(cv.MORPH_RECT, (1, 3))
    # 闭操作
    src_close = cv.morphologyEx(src_bin, cv.MORPH_CLOSE, kernel)
    image, contours, h = cv.findContours(src_close, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)

    minrect = []

    for i in range(len(contours)):
        rect = cv.minAreaRect(contours[i])
        if (rect[1][0] > (src.shape[1] * 0.5) and rect[1][1] > (src.shape[0] * 0.3) and rect[1][0] < (
                src.shape[1] * 0.8) and rect[1][1] < (src.shape[0] * 0.5)):
            minrect = cv.minAreaRect(contours[i])
            points = cv.boxPoints(minrect)  # 矩形四个点
    # 获得变换角度
    angle = minrect[2]
    # 获得变换矩阵
    change = cv.getRotationMatrix2D(minrect[0], angle, 1)
    src_correct = cv.warpAffine(src, change, (src.shape[1], src.shape[0]))
    src_gray = cv.warpAffine(src_gray, change, (src.shape[1], src.shape[0]))
    src_bin = cv.warpAffine(src_bin, change, (src.shape[1], src.shape[0]))
    # cv.imshow("src_correct",src_correct)

    array = np.array([[0, 0, 1]])
    newchange = np.vstack((change, array))
    # 根据变换矩阵得到新的四个点
    newpoints = []
    for i in range(4):
        point = newchange.dot(np.array([points[i][0], points[i][1], 1]))
        point = list(point)
        point.pop()
        newpoints.append(point)
    src_gray = src_gray[int(newpoints[1][1]) + 10:int(newpoints[0][1] - 10),
               int(newpoints[1][0]) + 10:int(newpoints[2][0]) - 10]
    angle = 0
    while (True):
        linepoint = line_detect(src_gray, low_thres, high_thres)  # 变成了一维数组
        Aline = []
        for line in linepoint:
            x1, y1, x2, y2 = line
            k1 = (y2 - y1) / ((x2 - x1) + 0.0000001)
            b1 = y1 - k1 * x1
            x = (src_gray.shape[0] - b1) / (k1 + 0.0000001)
            if (levelAngle(x1, y1, x2, y2) > 30 and levelAngle(x1, y1, x2, y2) < 150 and x < (
                    src_gray.shape[1] / 1.5) and x > (src_gray.shape[1] / 2.5)):
                Aline.append(line)
        if (len(Aline) == 1 or low_thres == 100):
            # print("%f度" % levelAngle(Aline[0][0], Aline[0][1], Aline[0][2], Aline[0][3]))
            if (len(Aline) == 0):
                break
            else:
                angle = levelAngle(Aline[0][0], Aline[0][1], Aline[0][2], Aline[0][3])
                # cv.line(src_gray, (Aline[0][0], Aline[0][1]), (Aline[0][2], Aline[0][3]), 0, 2)
                break
        else:
            low_thres += 1
    # 指针检测及角度计算完成

    # 数字的读取
    src = src[0:src.shape[0] // 2, :]
    src_gray = cv.cvtColor(src, cv.COLOR_BGR2GRAY)
    src_gray = cv.equalizeHist(src_gray)
    thres, src_bin = cv.threshold(src_gray, 100, 255, cv.THRESH_BINARY)
    image, contours, h = cv.findContours(src_bin, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
    for i in range(len(contours)):
        rect = cv.minAreaRect(contours[i])
        if (rect[1][0] > (src.shape[1] * 0.2) and rect[1][1] > (src.shape[0] * 0.2) and rect[1][0] < (
                src.shape[1] * 0.3) and
                rect[1][1] < (src.shape[0] * 0.4)):
            minrect = cv.minAreaRect(contours[i])
            points = cv.boxPoints(minrect)  # 矩形四个点
    array = np.array([[0, 0, 1]])
    newchange = np.vstack((change, array))
    # 根据变换矩阵得到新的四个点
    newpoints = []
    for i in range(4):
        point = newchange.dot(np.array([points[i][0], points[i][1], 1]))
        point = list(point)
        point.pop()
        newpoints.append(point)
    src_bin = cv.warpAffine(src_bin, change, (src.shape[1], src.shape[0]))
    src_num = src_bin[int(newpoints[1][1]) + 5:int(newpoints[0][1] - 5),
              int(newpoints[1][0]) + 10:int(newpoints[2][0]) - 5]
    # 进行数字分割
    _, src_num_open = cv.threshold(src_num, 50, 255, cv.THRESH_BINARY)
    src_num_open = cv.resize(src_num_open, (0, 0), fx=1, fy=0.8)
    model = load_model('num_model.h5')
    Num = ''
    for i in range(2):
        number = src_num_open[:, i * src_num_open.shape[1] // 2:(i + 1) * src_num_open.shape[1] // 2]
        mask = np.zeros((28, 28))
        number_row = 0
        number_col = 0
        if (number.shape[0] > 28):
            number_row = (number.shape[0] - 28) // 2
            if (number.shape[1] > 28):
                number_col = (number.shape[1] - 28) // 2
                number = number[number_row:number_row + 28, number_col:number_col + 28]
            else:
                for i in range(28):
                    for j in range(number.shape[1]):
                        mask[i][j + number_col] = number[number_row + i][j]
                number = mask
        else:
            number_row = (28 - number.shape[0]) // 2
            if (number.shape[1] < 28):
                number_col = (28 - number.shape[1]) // 2
                for i in range(number.shape[0]):
                    for j in range(number.shape[1]):
                        mask[i + number_row][j + number_col] = number[i][j]
                number = mask
            else:
                for i in range(number.shape[0]):
                    for j in range(28):
                        mask[i + number_row][j] = number[i][j + number_col]
                number = mask
        number = np.expand_dims(number, axis=0)
        number = np.expand_dims(number, axis=3)
        result = model.predict(number)
        max = -float('inf')
        num = 0
        for i in range(len(result[0])):
            if (max < result[0][i]):
                max = result[0][i]
                num = i
        Num += repr(num)

    return angle, Num
