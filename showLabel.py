import cv2
from Common import *


def showLabel(image, info):
    start = np.array([info["startPoint"]["x"], info["startPoint"]["y"]])
    end = np.array([info["endPoint"]["x"], info["endPoint"]["y"]])
    center = np.array([info["centerPoint"]["x"], info["centerPoint"]["y"]])
    meter = meterFinderByTemplate(image, info["template"])
    cv2.circle(meter, (start[0], start[1]), 5, (0, 0, 255), -1)
    cv2.circle(meter, (end[0], end[1]), 5, (0, 0, 255), -1)
    cv2.circle(meter, (center[0], center[1]), 5, (0, 0, 255), -1)
    cv2.imshow("your label", meter)
    cv2.waitKey(0)

