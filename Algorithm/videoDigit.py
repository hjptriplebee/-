from collections import defaultdict

import cv2
import numpy as np
import torch

from Algorithm.OCR.character.characterNet import characterNet
from Algorithm.pressure.digitPressure import digitPressure
from Algorithm.utils.Finder import meterFinderBySIFT


def videoDigit(video, info):
    """
    :param video: VideoCapture Input
    :param info: info
    :return: 
    """
    net = characterNet()
    net.load_state_dict(torch.load("Algorithm/OCR/character/NewNet_minLoss_model_0.965.pkl"))
    pictures = getPictures(video)  # 获得视频的帧，有少量重复帧

    def emptyLsit():
        return []

    imagesDict = defaultdict(emptyLsit)

    for i, frame in enumerate(pictures):
        res = digitPressure(frame, info)
        index = checkFrame(net, frame, info)
        imagesDict[chr(index+ord('A'))] += [res]
        # debug
        # title = str(chr(index+ord('A'))) + str(res)
        # frame = cv2.putText(frame, title, (50, 100), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 0, 255), 3)
        # cv2.imwrite(os.path.join("video_result", info['name']+str(i)+'.jpg'), frame)
        # cv2.imshow(title, frame)
        # cv2.waitKey(0)
    return imagesDict


def getPictures(videoCapture):
    """
    截取视频中每隔固定时间出现的帧
    :param videoCapture: 输入视频
    :return: 截取的帧片段
    """
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


def checkFrame(net, image, info):
    """
    判断图片的类型A，AB等
    :param image:  image
    :param info: info
    :return: 出现次序0.1.2.3
    """
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

    template = meterFinderBySIFT(image, info)
    template = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY)
    template = cv2.equalizeHist(template)
    dst = cv2.warpPerspective(template, M, (width, height))

    imgType = dst[characSplit[1][0]:characSplit[1][1], characSplit[0][0]:characSplit[0][1]]
    imgType = cv2.adaptiveThreshold(imgType, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 17, 11)
    imgType = cv2.bitwise_not(imgType)

    # orimage = imgType.copy()

    imgType = torch.Tensor(np.array(imgType, dtype=np.float32))
    imgType = torch.unsqueeze(imgType, 0)
    imgType = torch.unsqueeze(imgType, 0)
    type_probe = net.forward(imgType)
    type_probe = type_probe.detach().numpy()
    maxIndex = np.argmax(type_probe)

    # debug
    # cv2.imshow(str(chr(maxIndex+ord('A'))), orimage)
    # type = str(chr(maxIndex+ord('A')))
    # cv2.imwrite(os.path.join("video_result", type+info['name']+'_'+str(type_probe)+'.jpg'), orimage)
    # cv2.waitKey(0)

    return maxIndex
