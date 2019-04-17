import cv2
import numpy as np

from configuration import *
from Algorithm.utils.AngleFactory import AngleFactory


def getPoints(center, radious, angle):
    res = []
    farthestPointX = int(center[0] + radious * np.cos(angle / 180 * np.pi))
    farthestPointY = int(center[1] + radious * np.sin(angle / 180 * np.pi))

    for ro in range(radious // 3, radious):
        for theta in range(angle - 2, angle + 2):
            angleTemp = theta / 180 * np.pi
            x, y = int(ro * np.cos(angleTemp)) + center[0], int(ro * np.sin(angleTemp)) + center[1]
            res.append([x, y])
    return res, [farthestPointX, farthestPointY]


def EuclideanDistance(pt1, pt2):
    return np.sqrt((pt1[0] - pt2[0]) ** 2 + (pt1[1] - pt2[1]) ** 2)


def scanPointer(meter, info):
    """
    find pointer of meter
    :param meter: meter matched template
    :param pts: a list including three numpy array, eg: [startPointer, endPointer, centerPointer]
    :param startVal: an integer of meter start value
    :param endVal: an integer of meter ending value
    :return: pointer reading number
    """
    center = np.array([info["centerPoint"]["x"], info["centerPoint"]["y"]])
    start = np.array([info["startPoint"]["x"], info["startPoint"]["y"]])
    end = np.array([info["endPoint"]["x"], info["endPoint"]["y"]])
    startVal = info["startValue"]
    endVal = info["totalValue"]
    if meter.shape[0] > 500:
        fixHeight = 300
        fixWidth = int(meter.shape[1] / meter.shape[0] * fixHeight)
        resizeCoffX = fixWidth / meter.shape[1]
        meter = cv2.resize(meter, (fixWidth, fixHeight))

        start = (start * resizeCoffX).astype(np.int32)
        end = (end * resizeCoffX).astype(np.int32)
        center = (center * resizeCoffX).astype(np.int32)

    radious = int(EuclideanDistance(start, center))

    src = cv2.GaussianBlur(meter, (3, 3), sigmaX=0, sigmaY=0, borderType=cv2.BORDER_DEFAULT)

    gray = cv2.cvtColor(src=src, code=cv2.COLOR_RGB2GRAY)

    thresh = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, 17, 11)

    mask = np.zeros((src.shape[0], src.shape[1]), np.uint8)
    cv2.circle(mask, (center[0], center[1]), radious, (255, 255, 255), -1)
    thresh = cv2.bitwise_and(thresh, mask)
    cv2.circle(thresh, (center[0], center[1]), int(radious / 3), (0, 0, 0), -1)

    thresh = cv2.ximgproc.thinning(thresh, thinningType=cv2.ximgproc.THINNING_ZHANGSUEN)

    startAngle = int(
        AngleFactory.calAngleClockwise(startPoint=np.array([center[0] + 100, center[1]]), centerPoint=center,
                                       endPoint=start) * 180 / np.pi)
    endAngle = int(AngleFactory.calAngleClockwise(startPoint=np.array([center[0] + 100, center[1]]), centerPoint=center,
                                                  endPoint=end) * 180 / np.pi)
    # print(startAngle, endAngle)
    if endAngle <= startAngle:
        endAngle += 360
    maxSum = 0
    outerPoint = start
    for angle in range(startAngle - 10, endAngle + 10):
        pts, outPt = getPoints(center, radious, angle)
        thisSum = 0
        showImg = cv2.cvtColor(thresh.copy(), cv2.COLOR_GRAY2BGR)

        for pt in pts:
            cv2.circle(showImg, (pt[0], pt[1]), 2, (0, 0, 255), -1)
            if thresh[pt[1], pt[0]] != 0:
                thisSum += 1

        # cv2.circle(showImg, (outPt[0], outPt[1]), 2, (255, 0, 0), -1)
        # cv2.imshow("img", showImg)
        # cv2.waitKey(1)
        if thisSum > maxSum:
            maxSum = thisSum
            outerPoint = outPt

    if start[0] == outerPoint[0] and start[1] == outerPoint[1]:
        degree = startVal
    elif end[0] == outerPoint[0] and end[1] == outerPoint[1]:
        degree = endVal
    else:
        if start[0] == end[0] and start[1] == end[1]:
            end[0] -= 1
            end[1] -= 3
        degree = AngleFactory.calPointerValueByOuterPoint(start, end, center, outerPoint, startVal, endVal)

    # small value to zero
    if degree - startVal < 0.05 * (endVal - startVal):
        degree = startVal

    if ifShow:
        print("degree {:.2f} startPoint {}, endPoint{}, outPoint {}".format(degree, start, center, outerPoint))
        cv2.circle(meter, (outerPoint[0], outerPoint[1]), 10, (0, 0, 255), -1)
        cv2.line(meter, (center[0], center[1]), (outerPoint[0], outerPoint[1]), (0, 0, 255), 5)
        cv2.line(meter, (center[0], center[1]), (start[0], start[1]), (255, 0, 0), 3)
        cv2.line(meter, (center[0], center[1]), (end[0], end[1]), (255, 0, 0), 3)

        thresh = np.expand_dims(thresh, axis=2)
        thresh = np.concatenate((thresh, thresh, thresh), 2)
        meter = np.hstack((meter, thresh))

        cv2.imshow("test", meter)
        cv2.waitKey(0)
    return int(degree*100)/100
