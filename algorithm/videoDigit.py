import cv2
import sys
import numpy as np
import torch

from algorithm.Common import meterFinderBySIFT
from algorithm.pressure.digitPressure import digitPressure
from algorithm.OCR.svm.LeNet import LeNet

sys.path.append("../")

def videoDigit(video, info):
    """
    :param video: video
    :param info: info
    :return: 
    """
    global num
    result = {}

    start = ([info["startPoint"]["x"], info["startPoint"]["y"]])
    end = ([info["endPoint"]["x"], info["endPoint"]["y"]])
    center = ([info["centerPoint"]["x"], info["centerPoint"]["y"]])
    width = info["rectangle"]["width"]
    height = info["rectangle"]["height"]
    widthSplit = info["widthSplit"]
    heightSplit = info["heightSplit"]
    characSplit = info["characSplit"]

    # 计算数字表的矩形外框，并且拉直矫正
    fourth = (start[0] + end[0] - center[0], start[1] + end[1] - center[1])
    pts1 = np.float32([start, center, end, fourth])
    pts2 = np.float32([[0, 0], [width, 0], [width, height], [0, height]])
    M = cv2.getPerspectiveTransform(pts1, pts2)

    # 加载模型
    # clf = joblib.load("algorithm/OCR/svm/svm_model.m")
    net = LeNet()
    net.load_state_dict(torch.load("algorithm/OCR/svm/character_22net.pkl"))

    pictures = getPictures(video)  # 获得视频的帧，有少量重复帧
    imageDict = {}
    predicts = {}
    for i, frame in enumerate(pictures):
        template = meterFinderBySIFT(frame, info)
        template = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY)
        template = cv2.equalizeHist(template)
        dst = cv2.warpPerspective(template, M, (width, height))

        imgType = dst[characSplit[1][0]:characSplit[1][1], characSplit[0][0]:characSplit[0][1]]
        imgType = cv2.resize(imgType, (28, 28), interpolation=cv2.INTER_CUBIC)  # LeNet
        imgType = cv2.adaptiveThreshold(imgType, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 17, 11)

        # # svm模型
        # hog_vec, hog_vis = feature.hog(imgType, orientations=8, pixels_per_cell=(16, 16),
        #                                cells_per_block=(1, 1), block_norm='L2', visualise=True)
        # hog_vec = hog_vec.reshape(1, -1)
        # type_probe = clf.predict_proba(hog_vec)

        # LeNet 模型
        imgType = torch.Tensor(np.array(imgType, dtype=np.float32))
        imgType = torch.unsqueeze(imgType, 0)
        imgType = torch.unsqueeze(imgType, 0)
        type_probe = net.forward(imgType)
        type_probe = type_probe.detach().numpy()

        # find max probability frame
        maxIndex = np.argmax(type_probe)
        maxs = np.max(type_probe)
        # print(i, imgType.shape)
        # print("predict: ", type_probe)
        if maxIndex not in predicts.keys():
            imageDict[maxIndex] = frame
            predicts[maxIndex] = maxs
        elif predicts[maxIndex] < maxs:
            imageDict[maxIndex] = frame
        # cv2.imshow(str(maxIndex), imgType)
        # cv2.waitKey(0)
    # read meter
    # print(predicts)
    # print(imageDict)
    for k, frame in imageDict.items():
        result[chr(int(ord("A")+k))] = digitPressure(frame, info)
    return result


def getPictures(videoCapture):
    pictures = []
    cnt = 0
    skipFrameNum = 30
    while True:
        ret, frame = videoCapture.read()
        # print(cnt, np.shape(frame))
        cnt += 1
        if frame is None:
            break
        if cnt % skipFrameNum:
            continue
        pictures.append(frame)

    videoCapture.release()
    return pictures
