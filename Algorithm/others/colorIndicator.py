import cv2
import numpy as np

from Algorithm.utils.Finder import meterFinderByTemplate, meterFinderBySIFT

def colorIndicator(ROI, info):
    res = False
    image = meterFinderBySIFT(ROI, info)
    HSV = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    # color = [np.array([26, 43, 46]), np.array([34, 255, 255])]
    color = [np.array([11, 43, 46]), np.array([34, 255, 255])]
    Lower = color[0]
    Upper = color[1]
    mask = cv2.inRange(HSV, Lower, Upper)
    upmask = mask[int(0.25*mask.shape[0]):int(0.5*mask.shape[0]), :]
    upmask = cv2.bitwise_and(np.ones(upmask.shape, np.uint8), upmask)
    if np.sum(upmask) / upmask.shape[0]*upmask.shape[1] > 0.2:
        res = True
    return res