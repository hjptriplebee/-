import cv2
import numpy as np

from Algorithm.utils.Finder import meterFinderByTemplate
from Algorithm.utils.ScanPointer import EuclideanDistance


def colorPressure(image, info):
    """
    :param image: ROI image
    :param info: information for meter
    :return: value
    """
    meter = meterFinderByTemplate(image, info["template"])
    # get info
    startPoint = (info["startPoint"]["x"], info["startPoint"]["y"])
    # endPoint = (info["endPoint"]["x"], info["endPoint"]["y"])
    centerPoint = (info["centerPoint"]["x"], info["centerPoint"]["y"])
    
    # resize meter
    h, w, _ = meter.shape
    width = 300
    height = int(h * 300 / w)
    meter = cv2.resize(meter, (width, height))
    rateX = width / w
    rateY = height / h
    startPoint = (int(startPoint[0] * rateX), int(startPoint[1] * rateY))
    # endPoint = (int(endPoint[0] * rateX), int(endPoint[1] * rateY))
    centerPoint = (int(centerPoint[0] * rateX), int(centerPoint[1] * rateY))
    
    # 去掉表盘以外区域
    circleMask = cv2.bitwise_not(np.zeros(meter.shape, np.uint8))
    R = EuclideanDistance(centerPoint, startPoint)
    cv2.circle(circleMask, centerPoint, int(R), (0, 0, 0), -1)
    meter = cv2.bitwise_or(meter, circleMask)
    hsv = cv2.cvtColor(meter, cv2.COLOR_BGR2HSV)

    # 找到黑色指针区域
    lowerBlack = np.array([0, 0, 0], dtype="uint8")
    upperBlack = np.array([180, 255, 46], dtype="uint8")
    blackMask = cv2.inRange(hsv, lowerBlack, upperBlack)
    
    # 找到绿色指针区域
    lowerGreen = np.array([35, 43, 46], dtype="uint8")
    upperGreen = np.array([77, 255, 255], dtype="uint8")
    greenMask = cv2.inRange(hsv, lowerGreen, upperGreen)
    
    # 黑色区域和绿色区域求交集
    intersec = cv2.bitwise_and(blackMask, greenMask)
    result = np.sum(intersec)
    
    # cv2.imshow("blackMask", blackMask)
    # cv2.imshow("greenMask", greenMask)
    # cv2.imshow("intersec", intersec)
    # cv2.waitKey(0)
    return "green" if result / 255 > 3   else "red"
