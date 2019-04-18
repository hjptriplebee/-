"""
Created on Fri Jan  4 21:56:11 2019
@author: maoyingxue
"""

from Algorithm.utils.Finder import *


def countArrester(image, info):
    meter = meterFinderByTemplate(image, info["template"])

    image = meter
    startPoint=[info["startPoint"]["x"],info["startPoint"]["y"]]
    centerPoint=[info["centerPoint"]["x"],info["centerPoint"]["y"]]
    x_ratio=500.0/image.shape[1]
    y_ratio=500.0/image.shape[0]
    ratio=[x_ratio,y_ratio]
    startPoint = list(map(lambda x: int(x[0]*x[1]), zip(startPoint, ratio)))
    centerPoint = list(map(lambda x: int(x[0]*x[1]), zip(centerPoint, ratio)))

    image=cv2.resize(image,(500,500))
    Img=image
    R=Img.shape[0]//2

    ROIMask = cv2.bitwise_not(np.zeros(Img.shape, dtype = "uint8"))
    tableEdge=110
    cv2.circle(ROIMask, (R, R), R - tableEdge, (0, 0, 0), -1)
    Img = cv2.bitwise_or(Img, ROIMask)

    HSV = cv2.cvtColor(Img, cv2.COLOR_BGR2HSV)

    #red end line
    lowerRed = np.array([156, 43, 46], dtype="uint8")
    upperRed = np.array([180, 255, 255], dtype="uint8")
    redMask1 = cv2.inRange(HSV, lowerRed, upperRed)
    lowerRed = np.array([0, 20, 46], dtype="uint8")
    upperRed = np.array([10, 255, 255], dtype="uint8")
    redMask2 = cv2.inRange(HSV, lowerRed, upperRed)
    redMask=cv2.bitwise_or(redMask1,redMask2)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (6, 6))
    redMask = cv2.erode(redMask, kernel)
    redMask = cv2.dilate(redMask, kernel)

    _, redContours, redHierarchy = cv2.findContours(redMask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    redContours = sorted(redContours, key=lambda c: c.shape[0], reverse=True)
    redContours = [c for c in redContours if len(c) > 5 ]
    maxDis = 0
    pointerPoint = []

    for c in redContours:
        for p in c:
            dis = (p[0][0]-centerPoint[0])**2+(p[0][1]-centerPoint[1])**2
            if dis > maxDis:
                maxDis = dis
                pointerPoint = p[0]

    if ifShow:
        cv2.line(image, (centerPoint[0], centerPoint[1]), (pointerPoint[0], pointerPoint[1]), (0, 0, 255), 2)
        cv2.imshow("test", image)
        cv2.waitKey(0)

    # value = calAngleBetweenTwoVector(np.array(startPoint)-centerPoint,pointerPoint-centerPoint)
    if len(pointerPoint) == 0:
        return "pointer not found!"
    value = calAngleClockwise(np.array(startPoint), pointerPoint, centerPoint)
    value = int((value/(2*math.pi))*info["totalValue"])
    return value


def calAngleBetweenTwoVector(vectorA, vectorB):
    lenA = np.sqrt(vectorA.dot(vectorA))
    lenB = np.sqrt(vectorB.dot(vectorB))
    cosAngle = vectorA.dot(vectorB) / (lenA * lenB)
    angle = np.arccos(cosAngle)
    return angle


def calAngleClockwise( startPoint, endPoint, centerPoint):
    vectorA = startPoint - centerPoint
    vectorB = endPoint - centerPoint
    angle = calAngleBetweenTwoVector(vectorA, vectorB)
    # if counter-clockwise
    if np.cross(vectorA, vectorB) < 0:
        angle = 2 * np.pi - angle
    return angle
# if __name__ == '__main__':
#     file = open("../../config/" +"shw2_1.json")
#     info = json.load(file)
#     info["template"] = cv2.imread("../../template/shw2_1.jpg")
#     image = cv2.imread("../../image/shw2.jpg")
#     value = countArrester(image, info)
#     print(value)
#     print(value)
