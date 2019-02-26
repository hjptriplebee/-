from algorithm.Common import meterFinderBySIFT
from algorithm.debug import *

import cv2
import numpy as np


def onoffBatteryHardCode(image, info):
    """
    识别电池屏内部的开关是向上还是向下
    :param image: image
    :param info: information
    :return:
    """
    meter = meterFinderBySIFT(image, info)
    # print(meter.shape)
    meter = cv2.cvtColor(meter, cv2.COLOR_BGR2GRAY)
    meter = cv2.resize(meter, (200, 200))

    # 截取中心区域
    center = meter[60:120, 85:115]

    # 自适应二值化
    ret, thresh = cv2.threshold(center, 0, 255, cv2.THRESH_OTSU)

    # 比较上下区域的色度和，若上部区域比下部区域小，则说明开关是向上打开
    upRegion = np.sum(meter[:30])
    downRegion = np.sum(meter[30:])

    if ifShow:
        imgShow = np.hstack((center, thresh))
        cv2.imshow("ret", imgShow)
        cv2.waitKey(0)
        print(upRegion, downRegion)

    return 1 if upRegion < downRegion else 0


def onoffBattery(image, info):
    """
    识别电池屏内部的开关是向上还是向下
    :param image:
    :param info:
    :return:
    """
    meter = meterFinderBySIFT(image, info)
    gray = cv2.cvtColor(meter, cv2.COLOR_BGR2GRAY)

    start = ([info["startPoint"]["x"], info["startPoint"]["y"]])
    end = ([info["endPoint"]["x"], info["endPoint"]["y"]])
    center = ([info["centerPoint"]["x"], info["centerPoint"]["y"]])

    # width = info["rectangle"]["width"]
    # height = info["rectangle"]["height"]

    width = 40
    height = 80

    fourth = (start[0] + end[0] - center[0], start[1] + end[1] - center[1])
    pts1 = np.float32([start, center, end, fourth])
    pts2 = np.float32([[0, 0], [width, 0], [width, height], [0, height]])
    M = cv2.getPerspectiveTransform(pts1, pts2)
    dst = cv2.warpPerspective(gray, M, (width, height))

    # 自适应二值化
    ret, thresh = cv2.threshold(dst, 0, 255, cv2.THRESH_OTSU)
    # thresh = cv2.adaptiveThreshold(dst, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, 17, 11)

    # 比较上下区域的色度和，若上部区域比下部区域小，则说明开关是向上打开
    upRegion = np.mean(thresh[:40])
    downRegion = np.mean(thresh[40:])

    if ifShow:
        imgShow = np.hstack((dst, thresh))
        cv2.imshow("ret", imgShow)
        cv2.waitKey(0)
        print(upRegion, downRegion)

    return 1 if upRegion < downRegion else 0
