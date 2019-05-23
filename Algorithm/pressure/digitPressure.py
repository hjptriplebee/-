import os
import random

import cv2
import numpy as np

from Algorithm.OCR.utils import newNet
from Algorithm.utils.Finder import meterFinderBySIFT
from Algorithm.utils.boxRectifier import boxRectifier
from configuration import *


def digitPressure(image, info):
    template = meterFinderBySIFT(image, info)

    # 存储图片
    if not os.path.exists("storeDigitData"):
        os.mkdir("storeDigitData")

    try:
        os.mkdir("storeDigitData/thresh")
        os.mkdir("storeDigitData/rgb")
    except IOError:
        pass

    for i in range(11):
        try:
            os.mkdir("storeDigitData/thresh/" + str(i))
            os.mkdir("storeDigitData/rgb/" + str(i))
        except IOError:
            continue

    if 'rgb' in info and info['rgb']:  # rgb as input
        myRes = rgbRecognize(template, info)
    else:
        myRes = bitRecognize(template, info)

    if info["digitType"] == "KWH":
        myRes[0] = myRes[0][:4] + myRes.pop(1)

    # 去除头部的非数字字符，同时将非头部的字符转为数字
    for i in range(len(myRes)):
        temp = ""
        for j, c in enumerate(myRes[i]):
            if c != "?":
                temp += c
            elif j != 0:
                temp += str(random.randint(0, 9))
        myRes[i] = float(temp) if temp != "" else 0.0

    return myRes


def rgbRecognize(template, info):
    # 由标定点得到液晶区域
    dst = boxRectifier(template, info)
    # 读取标定信息
    widthSplit = info["widthSplit"]
    heightSplit = info["heightSplit"]
    # 网络初始化
    MyNet = newNet(if_rgb=True)
    myRes = []
    imgNum = int((len(os.listdir("storeDigitData/")) - 1) / 3)
    for i in range(len(heightSplit)):
        split = widthSplit[i]
        myNum = ""
        for j in range(len(split) - 1):
            if "decimal" in info.keys() and j == info["decimal"][i]:
                myNum += "."
                continue
            # 得到分割的图片区域
            img = dst[heightSplit[i][0]:heightSplit[i][1], split[j]:split[j + 1]]
            num = MyNet.recognizeNet(img)
            myNum = myNum + num

            # 存储图片
            cv2.imwrite("storeDigitData/rgb/{}/{}_{}{}_p{}.bmp".format(
                num, imgNum,  i, j,  num), img)
        myRes.append(myNum)

    if ifShow:
        cv2.imshow("rec", dst)
        cv2.imshow("template", template)
        print(myRes)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    return myRes


def bitRecognize(template, info):
    template = cv2.GaussianBlur(template, (3, 3), 0)

    # 读取标定信息
    widthSplit = info["widthSplit"]
    heightSplit = info["heightSplit"]

    # 将info中的参数加入代码中
    block = info["thresh"]["block"]
    param = info["thresh"]["param"]
    ifOpen = info["ifopen"]

    # 由标定点得到液晶区域
    dst = boxRectifier(template, info)

    # 灰度图
    gray = cv2.cvtColor(dst, cv2.COLOR_BGR2GRAY)

    # 针对不同的数字表类型进行不同的增强
    if info["digitType"] != "TTC":
        Blur = cv2.GaussianBlur(gray, (5, 5), 0)
        Hist = cv2.equalizeHist(Blur)
        thresh = cv2.adaptiveThreshold(Hist, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, 15, 11)
    else:
        thresh = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, block, param)
        if ifOpen == "close":
            p = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
            thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, p)

    if os.path.exists("storeDigitData/"):
        imgNum = int((len(os.listdir("storeDigitData/")) - 1) / 3)
        cv2.imwrite("storeDigitData/" + str(imgNum) + "_dst.bmp", dst)
        cv2.imwrite("storeDigitData/" + str(imgNum) + "_gray.bmp", gray)
        cv2.imwrite("storeDigitData/" + str(imgNum) + "_thresh.bmp", thresh)

    # 网络初始化
    MyNet = newNet(if_rgb=False)
    myRes = []
    imgNum = int((len(os.listdir("storeDigitData/")) - 1) / 3)

    for i in range(len(heightSplit)):
        split = widthSplit[i]
        myNum = ""
        for j in range(len(split) - 1):
            if "decimal" in info.keys() and j == info["decimal"][i]:
                myNum += "."
                continue
            # 得到分割的图片区域
            img = thresh[heightSplit[i][0]:heightSplit[i][1], split[j]:split[j + 1]]
            rgb_ = dst[heightSplit[i][0]:heightSplit[i][1], split[j]:split[j + 1]]

            num = MyNet.recognizeNet(img)
            myNum = myNum + num

            # 存储图片
            cv2.imwrite("storeDigitData/thresh/{}/{}_{}{}_p{}.bmp".format(
                num, imgNum, i, j, num), img)
            cv2.imwrite("storeDigitData/rgb/{}/{}_{}{}_p{}.bmp".format(
                num, imgNum, i, j, num), rgb_)
        myRes.append(myNum)

    if ifShow:
        cv2.imshow("rec", dst)
        cv2.imshow("template", template)
        print(myRes)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    return myRes
