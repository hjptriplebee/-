# -*- coding: utf-8 -*-
"""
Created on Tue Apr 16 19:32:29 2019
@author: sun
"""
import cv2
import numpy as np
import math
import json
from Algorithm.utils.Finder import meterFinderBySIFT
from Algorithm.utils.boxRectifier import boxRectifier

# 获取图像的灰度矩阵
def getMatrix(img):
    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return gray_img


# 实现图像的卷积操作
def imgConvoluting(image, filter):
    w, h = filter.shape
    con_img = image.copy()
    filter_w, filter_h = filter.shape
    for i in range(1, len(image) + 1 - w):
        for j in range(1, len(image[i]) + 1 - h):
            con_img[i][j] = (image[i:i + filter_w:, j:j + filter_h:] * filter).sum()

    return con_img


# 使用二阶微分实现检测，实现图像的锐化
def imgEnhance(image, c):
    raw_img = image.copy()
    res_img = image.copy()

    image = cv2.GaussianBlur(image, (17, 17), 1)

    filter = np.array([[0, 1, 0], [1, -4, 1], [0, 1, 0]])

    filter_img = imgConvoluting(image, filter)

    for i in range(len(image)):
        for j in range(len(image[i])):
            res_img[i][j] = raw_img[i][j] + c * filter_img[i][j]

    return res_img


# 读取图像数据到矩阵
def readImg(filepath):
    if filepath == "":
        return None

    img = cv2.imread(filepath)
    return img


def hasCircle(img):
    '''
    :param img: 二值图，图像边缘信息
    :return:
    '''

    circles = cv2.HoughCircles(img, cv2.HOUGH_GRADIENT, 1, 50, param1=80, param2=30, minRadius=10, maxRadius=40)

    if circles is None:
        return "off"

    return "on"

def contactStatus(image, info):
    """
    :param image:whole image
    :param info:开关配置
    :return: on代表1，off代表0
    """
    template = meterFinderBySIFT(image, info)
    template = cv2.GaussianBlur(template, (3, 3), 0)

    image = boxRectifier(template, info)

    debug=False
    if debug:
        cv2.imshow("image", image)
        cv2.imshow("template",info['template'])
        cv2.waitKey(0)



    # 获取图像灰度矩阵
    image = getMatrix(image)
    # 边缘检测
    #image = cv2.Canny(image, 10, 200)
    #直接在灰度图上检测圆
    return hasCircle(image)
# if __name__ == "__main__":
#     # img = readImg("./case/case1.png")
#     #img = getMatrix("../images/meterReader/contact1_1.png")
#     img = cv2.imread("../../info/20190410/template/contact1_1.png")
#
#     # 图像边缘增强
#     #img = imgEnhance(img, 1)
#     #cv2.imshow("img", img)
#     # 边缘检测
#
#     # ret,th = cv2.threshold(img,10,255,cv2.THRESH_BINARY)
#     # ret,th = cv2.threshold(img,100,255,cv2.THRESH_BINARY) #case1与case3均适用
#
#     # print(img)
#
#
#     # Otsu 滤波
#     # ret2,th2 = cv2.threshold(img,0,100,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
#
#     # cv2.imwrite("gass.png", img)
#     # cv2.waitKey(0)
#     # cv2.destroyWindow()