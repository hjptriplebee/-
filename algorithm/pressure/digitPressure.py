import sys
import os

from algorithm.Common import *
from algorithm.OCR.utils import *
from algorithm.debug import *

sys.path.append("algorithm/OCR/LeNet")


def digitPressure(image, info):
    template = meterFinderBySIFT(image, info)
    template = cv2.GaussianBlur(template, (3, 3), 0)
    template = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY)

    # 读取标定信息
    start = ([info["startPoint"]["x"], info["startPoint"]["y"]])
    end = ([info["endPoint"]["x"], info["endPoint"]["y"]])
    center = ([info["centerPoint"]["x"], info["centerPoint"]["y"]])
    width = info["rectangle"]["width"]
    height = info["rectangle"]["height"]
    widthSplit = info["widthSplit"]
    heightSplit = info["heightSplit"]

    # 计算数字表的矩形外框，并且拉直矫正
    fourth = (start[0] + end[0] - center[0], start[1] + end[1] - center[1])
    pts1 = np.float32([start, center, end, fourth])
    pts2 = np.float32([[0, 0], [width, 0], [width, height], [0, height]])
    M = cv2.getPerspectiveTransform(pts1, pts2)
    dst = cv2.warpPerspective(template, M, (width, height))
    # dst = cv2.equalizeHist(dst)

    # 网络初始化
    MyNet = newNet()
    myRes = []

    for i in range(len(heightSplit)):
        split = widthSplit[i]
        myNum = ""
        for j in range(len(split) - 1):
            if "decimal" in info.keys() and j == info["decimal"][i]:
                myNum += "."
                continue
            img = dst[heightSplit[i][0]:heightSplit[i][1], split[j]:split[j + 1]]

            # cv2.imshow("debug3", img)
            if info["digitType"] != "TTC":
                img = cv2.GaussianBlur(img, (5, 5), 0)
                img = cv2.equalizeHist(img)
                # cv2.imshow("debug", img)
                img = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, 13, 11)
            elif info["digitType"] == "TTC":
                img = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 55, 11)
            # cv2.imshow("debug2", img)

            # imgNum = len(os.listdir("storeDigitData/"))
            # cv2.imwrite("storeDigitData/" + str(imgNum) + ".jpg", img)

            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 2))
            img = cv2.morphologyEx(img, cv2.MORPH_OPEN, kernel)

            num = MyNet.recognizeNet(img)

            myNum += num

        myRes.append(myNum)

    if info["digitType"] == "KWH":
        myRes[0] = myRes[0][:4]+myRes.pop(1)

    for i in range(len(myRes)):
        temp = ""
        for j, c in enumerate(myRes[i]):
            if c != "?":
                temp += c
            elif j != 0:
                temp += str(random.randint(0, 9))
        myRes[i] = temp

    K.clear_session()

    if ifShow:
        cv2.circle(template, (start[0], start[1]), 5, (0, 0, 255), -1)
        cv2.circle(template, (end[0], end[1]), 5, (0, 255, 0), -1)
        cv2.circle(template, (center[0], center[1]), 5, (255, 0, 0), -1)
        cv2.circle(template, (fourth[0], fourth[1]), 5, (255, 255, 0), -1)
        cv2.imshow("tem", template)
        cv2.imshow("rec", dst)
        print(myRes)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    return myRes
