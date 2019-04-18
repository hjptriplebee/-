import cv2
import numpy as np

from Algorithm.utils.ScanPointer import EuclideanDistance


def boxRectifier(templateImage, info):
    start = ([info["startPoint"]["x"], info["startPoint"]["y"]])
    end = ([info["endPoint"]["x"], info["endPoint"]["y"]])
    center = ([info["centerPoint"]["x"], info["centerPoint"]["y"]])
    if "rectangel" in info:
        width = info["rectangle"]["width"]
        height = info["rectangle"]["height"]
    else:
        width = int(EuclideanDistance(start, center))
        height = int(EuclideanDistance(center, end))

    # 计算数字表的矩形外框，并且拉直矫正
    fourth = (start[0] + end[0] - center[0], start[1] + end[1] - center[1])
    pts1 = np.float32([start, center, end, fourth])
    pts2 = np.float32([[0, 0], [width, 0], [width, height], [0, height]])
    M = cv2.getPerspectiveTransform(pts1, pts2)
    dst = cv2.warpPerspective(templateImage, M, (width, height))
    return dst
