from algorithm.Common import *
from configuration import *


def SF6Reader(image, info):
    """
    :param image: ROI image
    :param info: information for this meter
    :return: value
    """

    # get info
    startPoint = (info["startPoint"]["x"], info["startPoint"]["y"])
    endPoint = (info["endPoint"]["x"], info["endPoint"]["y"])
    centerPoint = (info["centerPoint"]["x"], info["centerPoint"]["y"])
    startValue = info["startValue"]
    totalValue = info["totalValue"]

    # template match
    # meter = (image, infcentero["template"])
    meter = meterFinderBySIFT(image, info)

    # resize to a fixed size
    fixHeight = 300
    fixWidth = int(meter.shape[1] / meter.shape[0] * fixHeight)
    resizeCoffX = fixWidth / meter.shape[1]
    resizeCoffY = fixHeight / meter.shape[0]
    meter = cv2.resize(meter, (fixWidth, fixHeight))
    startPoint = (int(startPoint[0] * resizeCoffX), int(startPoint[1] * resizeCoffY))
    endPoint = (int(endPoint[0] * resizeCoffX), int(endPoint[1] * resizeCoffY))
    centerPoint = (int(centerPoint[0] * resizeCoffX), int(centerPoint[1] * resizeCoffY))

    # hist equalization
    ROIYUV = cv2.cvtColor(meter, cv2.COLOR_BGR2YUV)
    ROIYUV[:, :, 0] = cv2.equalizeHist(ROIYUV[:, :, 0])
    meter = cv2.cvtColor(ROIYUV, cv2.COLOR_YUV2BGR)

    # circle mask
    ROIMask = cv2.bitwise_not(np.zeros(meter.shape, dtype="uint8"))
    R = min([centerPoint[0], centerPoint[1], meter.shape[1] - centerPoint[0], meter.shape[0] - centerPoint[1]])
    edge = 10
    cv2.circle(ROIMask, centerPoint, R - edge, (0, 0, 0), -1)
    meter = cv2.bitwise_or(meter, ROIMask)

    # black mask
    lowerBlack = np.array([0, 0, 0], dtype="uint8")
    upperBlack = np.array([180, 255, 46], dtype="uint8")
    ROIHSV = cv2.cvtColor(meter, cv2.COLOR_BGR2HSV)
    blackMask = cv2.inRange(ROIHSV, lowerBlack, upperBlack)

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    blackMask = cv2.erode(blackMask, kernel)
    blackMask = cv2.dilate(blackMask, kernel)
    blackMask = cv2.dilate(blackMask, kernel)
    # cv2.imshow("black", blackMask)

    # find contours
    _, blackContours, blackHierarchy = cv2.findContours(blackMask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    blackContours = sorted(blackContours, key=lambda c: c.shape[0], reverse=True)
    minPointNum = 50
    blackContours = [c for c in blackContours if len(c) > minPointNum]
    # cv2.imshow("blackMask", blackMask)

    # find the farthest point
    pointDis = 0
    point = []
    for c in blackContours:
        for p in c:
            dis = (p[0][0] - centerPoint[0]) * (p[0][0] - centerPoint[0]) + (p[0][1] - centerPoint[1]) * (
            p[0][1] - centerPoint[1])
            if dis > pointDis:
                pointDis = dis
                point = p[0]

    res = AngleFactory.calPointerValueByOuterPoint(np.array(startPoint), np.array(endPoint), np.array(centerPoint),
                                                   np.array(point), startValue, totalValue)

    if ifShow:
        cv2.line(meter, centerPoint, (point[0], point[1]), (0, 255, 0), 2)
        cv2.line(meter, centerPoint, startPoint, (0, 255, 0), 2)
        cv2.line(meter, centerPoint, endPoint, (0, 255, 0), 2)
        cv2.imshow("meter", meter)
        cv2.waitKey(0)
    return res
