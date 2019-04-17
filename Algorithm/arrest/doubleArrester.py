"""
@author: Xuyuanyuan
"""

import cv2
import numpy as np

from Algorithm.utils.Finder import meterFinderByTemplate
from Algorithm.utils.ScanPointer import scanPointer
from configuration import *


def doubleArrester(image, info):
    """
    :param image: ROI image
    :param info: information for this meter
    :return: value
    """
    meter = meterFinderByTemplate(image, info["template"])

    ovalCenter = np.array([info["centerPointUp"]["x"], info["centerPointUp"]["y"]])
    ovalStart = np.array([info["startPointUp"]["x"], info["startPointUp"]["y"]])
    ovalEnd = np.array([info["endPointUp"]["x"], info["endPointUp"]["y"]])
    rectCenter = np.array([info["centerPoint"]["x"], info["centerPoint"]["y"]])
    rectStart = np.array([info["startPoint"]["x"], info["startPoint"]["y"]])
    rectEnd = np.array([info["endPoint"]["x"], info["endPoint"]["y"]])
    h, w, _ = meter.shape
    result = {}

    # crop oval meter
    r = int(np.sqrt(np.sum(np.square(ovalCenter - ovalStart))) * 1.6)
    y = max(0, int(ovalCenter[1] - r))
    x = max(0, int(ovalCenter[0] - r))
    width = min(w, x + 2 * r)
    height = min(h, y + 2 * r)
    ovalCenter = np.array([ovalCenter[0] - x, ovalCenter[1] - y])
    ovalStart = np.array([ovalStart[0] - x, ovalStart[1] - y])
    ovalEnd = np.array([ovalEnd[0] - x, ovalEnd[1] - y])
    ovalMeter = meter[y:height, x:width]

    result["ovalMeter"] = scanPointer(ovalMeter, [ovalStart, ovalEnd, ovalCenter], info["startValueUp"],
                                      info["totalValueUp"])
    if ifShow:
        cv2.circle(ovalMeter, (ovalStart[0], ovalStart[1]), 5, (0, 0, 255), -1)
        cv2.circle(ovalMeter, (ovalCenter[0], ovalCenter[1]), 5, (0, 255, 0), -1)
        cv2.circle(ovalMeter, (ovalEnd[0], ovalEnd[1]), 5, (255, 0, 0), -1)
        cv2.imshow("rectMeter", ovalMeter)
        cv2.waitKey(0)
        print("ovalMeter", result["ovalMeter"])

    # crop rectangle meter
    r = int(np.sqrt(np.sum(np.square(rectCenter - rectStart))) * 1.6)
    y = max(0, int(rectCenter[1] - r))
    x = max(0, int(rectCenter[0] - r))
    width = min(w, x + 2 * r)
    height = min(h, y + 2 * r)
    rectCenter = np.array([rectCenter[0] - x, rectCenter[1] - y])
    rectStart = np.array([rectStart[0] - x, rectStart[1] - y])
    rectEnd = np.array([rectEnd[0] - x, rectEnd[1] - y])
    rectMeter = meter[y:height, x:width]
    result["rectMeter"] = scanPointer(rectMeter, [rectStart, rectEnd, rectCenter], info["startValue"],
                                      info["totalValue"])

    if ifShow:
        cv2.circle(rectMeter, (rectStart[0], rectStart[1]), 5, (0, 0, 255), -1)
        cv2.circle(rectMeter, (rectCenter[0], rectCenter[1]), 5, (0, 0, 255), -1)
        cv2.circle(rectMeter, (rectEnd[0], rectEnd[1]), 5, (0, 0, 255), -1)
        cv2.imshow("rectMeter", rectMeter)
        cv2.waitKey(0)
        print("rectMeter", result["rectMeter"])

    return result
