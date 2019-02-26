import json

import cv2
import numpy as np

from algorithm.Common import *

plot_index = 0

def normalPressure(image, info):
    """
    :param image: ROI image
    :param info: information for this meter
    :return: value
    """
    meter = meterFinderBySIFT(image, info)
    result = scanPointer(meter, info)
    return result

def inc():
    global plot_index
    plot_index += 1
    return plot_index

def calAvgRadius(center, end_ptr, radius, start_ptr):
    radius_1 = np.sqrt(np.power(start_ptr[0] - center[0], 2) + np.power(start_ptr[1] - center[1], 2))
    radius_2 = np.sqrt(np.power(end_ptr[0] - center[0], 2) + np.power(end_ptr[1] - center[1], 2))
    radius = np.int64((radius_1 + radius_2) / 2)
    return radius

def cvtPtrDic2D(dic_ptr):
    """
    point.x,point.y转numpy数组
    :param dic_ptr:
    :return:
    """
    if dic_ptr['x'] and dic_ptr['y'] is not None:
        dic_ptr = np.array([dic_ptr['x'], dic_ptr['y']])
    else:
        return np.array([0, 0])
    return dic_ptr

def cv2PtrTuple2D(tuple):
    """
    tuple 转numpy 数组
    :param tuple:
    :return:
    """
    if tuple[0] and tuple[1] is not None:
        tuple = np.array([tuple[0], tuple[1]])
    else:
        return np.array([0, 0])
    return tuple

def cleanNoisedRegions(src, info, shape):
    """
    根据标定信息清楚一些有干扰性的区域
    :param src:
    :param info:
    :param shape:
    :return:
    """
    if 'noisedRegion' in info and info['noisedRegion'] is not None:
        region_roi = info['noisedRegion']
        for roi in region_roi:
            mask = cv2.bitwise_not(np.zeros(shape=(shape[0], shape[1]), dtype=np.uint8))
            mask[roi[1]:roi[1] + roi[3], roi[0]:roi[0] + roi[2]] = 0
            src = cv2.bitwise_and(src, mask)
    return src
