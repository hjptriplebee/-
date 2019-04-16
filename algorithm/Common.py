import math

import cv2
import numpy as np
from sklearn.metrics.pairwise import pairwise_distances

from configuration import *


def meterFinderByTemplate(image, template):
    """
    locate meter's bbox
    :param image: image
    :param template: template
    :return: bbox image
    """
    methods = [cv2.TM_CCOEFF, cv2.TM_CCOEFF_NORMED, cv2.TM_CCORR,
               cv2.TM_CCORR_NORMED, cv2.TM_SQDIFF, cv2.TM_SQDIFF_NORMED]

    w, h, _ = template.shape

    # for test
    # cv2.imshow("test", img)
    # img = (img * 0.5).astype(np.uint8) # test
    # cv2.imshow("test2", img)
    # cv2.waitKey(0)

    i = 5
    res = cv2.matchTemplate(image, template, methods[i])
    minVal, maxVal, minLoc, maxLoc = cv2.minMaxLoc(res)
    if methods[i] in [cv2.TM_SQDIFF, cv2.TM_SQDIFF_NORMED]:
        topLeft = minLoc
    else:
        topLeft = maxLoc
    bottomRight = (topLeft[0] + h, topLeft[1] + w)

    return image[topLeft[1]:bottomRight[1], topLeft[0]:bottomRight[0]]


def meterLocationFinderBySIFT(image, template):
    """
    locate meter's bbox
    :param image: image
    :param template: template
    :return: bbox
    """
    templateBlurred = cv2.GaussianBlur(template, (3, 3), 0)
    imageBlurred = cv2.GaussianBlur(image, (3, 3), 0)

    sift = cv2.xfeatures2d.SIFT_create()

    # shape of descriptor n * 128, n is the num of key points.
    # a row of descriptor is the feature of related key point.
    templateKeyPoint, templateDescriptor = sift.detectAndCompute(templateBlurred, None)
    imageKeyPoint, imageDescriptor = sift.detectAndCompute(imageBlurred, None)

    # for debug
    # templateBlurred = cv2.drawKeypoints(templateBlurred, templateKeyPoint, templateBlurred)
    # imageBlurred = cv2.drawKeypoints(imageBlurred, imageKeyPoint, imageBlurred)
    # cv2.imshow("template", templateBlurred)
    # cv2.imshow("image", imageBlurred)

    # match
    bf = cv2.BFMatcher()
    # k = 2, so each match has 2 points. 2 points are sorted by distance.
    matches = bf.knnMatch(templateDescriptor, imageDescriptor, k=2)

    # The first one is better than the second one
    good = [[m] for m, n in matches if m.distance < 0.7 * n.distance]

    # distance matrix
    templatePointMatrix = np.array([list(templateKeyPoint[p[0].queryIdx].pt) for p in good])
    imagePointMatrix = np.array([list(imageKeyPoint[p[0].trainIdx].pt) for p in good])
    templatePointDistanceMatrix = pairwise_distances(templatePointMatrix, metric="euclidean")
    imagePointDistanceMatrix = pairwise_distances(imagePointMatrix, metric="euclidean")

    # del bad match
    distances = []
    maxAbnormalNum = 15
    for i in range(len(good)):
        diff = abs(templatePointDistanceMatrix[i] - imagePointDistanceMatrix[i])
        # distance between distance features
        diff.sort()
        distances.append(np.sqrt(np.sum(np.square(diff[:-maxAbnormalNum]))))

    averageDistance = np.average(distances)
    good2 = [good[i] for i in range(len(good)) if distances[i] < 2 * averageDistance]

    # for debug
    # matchImage = cv2.drawMatchesKnn(template, templateKeyPoint, image, imageKeyPoint, good2, None, flags=2)
    # cv2.imshow("matchImage", matchImage)
    # cv2.waitKey(0)

    matchPointMatrix = np.array([list(imageKeyPoint[p[0].trainIdx].pt) for p in good2])

    # for p1, p2 in matchPointMatrix:
    #     cv2.circle(image, (int(p1), int(p2)), 0, (255, 0, 0), thickness=50)
    #     print(p1, p2)
    # cv2.imshow("matchImage", image)

    minX = int(np.min(matchPointMatrix[:, 0]))
    maxX = int(np.max(matchPointMatrix[:, 0]))
    minY = int(np.min(matchPointMatrix[:, 1]))
    maxY = int(np.max(matchPointMatrix[:, 1]))
    return minX, minY, maxX, maxY


def meterFinderBySIFT(image, info):
    """
    locate meter's bbox
    :param image: image
    :param info: info
    :return: bbox image
    """
    template = info["template"]

    # cv2.imshow("template", template)
    # cv2.imshow("image", image)
    # cv2.waitKey(0)

    startPoint = (info["startPoint"]["x"], info["startPoint"]["y"])
    centerPoint = (info["centerPoint"]["x"], info["centerPoint"]["y"])
    endPoint = (info["endPoint"]["x"], info["endPoint"]["y"])
    # startPointUp = (info["startPointUp"]["x"], info["startPointUp"]["y"])
    # endPointUp = (info["endPointUp"]["x"], info["endPointUp"]["y"])
    # centerPointUp = (info["centerPointUp"]["x"], info["centerPointUp"]["y"])

    templateBlurred = cv2.GaussianBlur(template, (3, 3), 0)
    imageBlurred = cv2.GaussianBlur(image, (3, 3), 0)

    sift = cv2.xfeatures2d.SIFT_create()

    # shape of descriptor n * 128, n is the num of key points.
    # a row of descriptor is the feature of related key point.
    templateKeyPoint, templateDescriptor = sift.detectAndCompute(templateBlurred, None)
    imageKeyPoint, imageDescriptor = sift.detectAndCompute(imageBlurred, None)

    # for debug
    # templateBlurred = cv2.drawKeypoints(templateBlurred, templateKeyPoint, templateBlurred)
    # imageBlurred = cv2.drawKeypoints(imageBlurred, imageKeyPoint, imageBlurred)
    # # cv2.imshow("template", templateBlurred)
    # cv2.imshow("image", imageBlurred)
    # cv2.waitKey(0)

    # match
    bf = cv2.BFMatcher()
    # k = 2, so each match has 2 points. 2 points are sorted by distance.
    matches = bf.knnMatch(templateDescriptor, imageDescriptor, k=2)

    # The first one is better than the second one
    good = [[m] for m, n in matches if m.distance < 0.8 * n.distance]
    # distance matrix
    templatePointMatrix = np.array([list(templateKeyPoint[p[0].queryIdx].pt) for p in good])
    imagePointMatrix = np.array([list(imageKeyPoint[p[0].trainIdx].pt) for p in good])
    templatePointDistanceMatrix = pairwise_distances(templatePointMatrix, metric="euclidean")
    imagePointDistanceMatrix = pairwise_distances(imagePointMatrix, metric="euclidean")

    # del bad match
    distances = []
    maxAbnormalNum = 15
    for i in range(len(good)):
        diff = abs(templatePointDistanceMatrix[i] - imagePointDistanceMatrix[i])
        # distance between distance features
        diff.sort()
        distances.append(np.sqrt(np.sum(np.square(diff[:-maxAbnormalNum]))))

    averageDistance = np.average(distances)
    good2 = [good[i] for i in range(len(good)) if distances[i] < 2 * averageDistance]

    # for debug
    # matchImage = cv2.drawMatchesKnn(template, templateKeyPoint, image, imageKeyPoint, good2, None, flags=2)
    # cv2.imshow("matchImage", matchImage)
    # cv2.waitKey(0)

    # not match
    if len(good2) < 3:
        print("not found!")
        return template

    # 寻找转换矩阵 M
    src_pts = np.float32([templateKeyPoint[m[0].queryIdx].pt for m in good2]).reshape(-1, 1, 2)
    dst_pts = np.float32([imageKeyPoint[m[0].trainIdx].pt for m in good2]).reshape(-1, 1, 2)
    M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
    matchesMask = mask.ravel().tolist()
    h, w, _ = template.shape

    # 找出匹配到的图形的四个点和标定信息里的所有点
    pts = np.float32(
        [[0, 0], [0, h - 1], [w - 1, h - 1], [w - 1, 0], [startPoint[0], startPoint[1]], [endPoint[0], endPoint[1]],
         [centerPoint[0], centerPoint[1]],
         # [startPointUp[0], startPointUp[1]],
         # [endPointUp[0], endPointUp[1]],
         # [centerPointUp[0], centerPointUp[1]]
         ]).reshape(-1, 1, 2)
    dst = cv2.perspectiveTransform(pts, M)

    # 校正图像
    angle = 0.0
    vector = (dst[3][0][0] - dst[0][0][0], dst[3][0][1] - dst[0][0][1])
    cos = (vector[0] * (200.0)) / (200.0 * math.sqrt(vector[0] ** 2 + vector[1] ** 2))
    if (vector[1] > 0):
        angle = math.acos(cos) * 180.0 / math.pi
    else:
        angle = -math.acos(cos) * 180.0 / math.pi
    # print(angle)

    change = cv2.getRotationMatrix2D((dst[0][0][0], dst[0][0][1]), angle, 1)
    src_correct = cv2.warpAffine(image, change, (image.shape[1], image.shape[0]))
    array = np.array([[0, 0, 1]])
    newchange = np.vstack((change, array))
    # 获得校正后的所需要的点
    newpoints = []
    for i in range(len(pts)):
        point = newchange.dot(np.array([dst[i][0][0], dst[i][0][1], 1]))
        point = list(point)
        point.pop()
        newpoints.append(point)
    src_correct = src_correct[int(round(newpoints[0][1])):int(round(newpoints[1][1])),
                  int(round(newpoints[0][0])):int(round(newpoints[3][0]))]

    width = src_correct.shape[1]
    height = src_correct.shape[0]
    if width == 0 or height == 0:
        return template

    startPoint = (int(round(newpoints[4][0]) - newpoints[0][0]), int(round(newpoints[4][1]) - newpoints[0][1]))
    endPoint = (int(round(newpoints[5][0]) - newpoints[0][0]), int(round(newpoints[5][1]) - newpoints[0][1]))
    centerPoint = (int(round(newpoints[6][0]) - newpoints[0][0]), int(round(newpoints[6][1]) - newpoints[0][1]))

    def isOverflow(point, width, height):
        if point[0] < 0 or point[1] < 0 or point[0] > width - 1 or point[1] > height - 1:
            return True
        return False

    if isOverflow(startPoint, width, height) or isOverflow(endPoint, width, height) or isOverflow(centerPoint, width,
                                                                                                  height):
        print("overflow!")
        return template

    # startPointUp = (int(round(newpoints[7][0]) - newpoints[0][0]), int(round(newpoints[7][1]) - newpoints[0][1]))
    # endPointUp = (int(round(newpoints[8][0]) - newpoints[0][0]), int(round(newpoints[8][1]) - newpoints[0][1]))
    # centerPointUp = (int(round(newpoints[9][0]) - newpoints[0][0]), int(round(newpoints[9][1]) - newpoints[0][1]))
    info["startPoint"]["x"] = startPoint[0]
    info["startPoint"]["y"] = startPoint[1]
    info["centerPoint"]["x"] = centerPoint[0]
    info["centerPoint"]["y"] = centerPoint[1]
    info["endPoint"]["x"] = endPoint[0]
    info["endPoint"]["y"] = endPoint[1]

    return src_correct


class AngleFactory:
    """method for angle calculation"""

    @staticmethod
    def __calAngleBetweenTwoVector(vectorA, vectorB):
        """
        get angle formed by two vector
        :param vectorA: vector A
        :param vectorB: vector B
        :return: angle
        """
        lenA = np.sqrt(vectorA.dot(vectorA))
        lenB = np.sqrt(vectorB.dot(vectorB))
        cosAngle = vectorA.dot(vectorB) / (lenA * lenB)
        angle = np.arccos(cosAngle)
        return angle

    @classmethod
    def calAngleClockwise(cls, startPoint, endPoint, centerPoint):
        """
        get clockwise angle formed by three point
        :param startPoint: start point
        :param endPoint: end point
        :param centerPoint: center point
        :return: clockwise angle
        """
        vectorA = startPoint - centerPoint
        vectorB = endPoint - centerPoint
        angle = cls.__calAngleBetweenTwoVector(vectorA, vectorB)

        # if counter-clockwise
        if np.cross(vectorA, vectorB) < 0:
            angle = 2 * np.pi - angle

        return angle

    @classmethod
    def calPointerValueByOuterPoint(cls, startPoint, endPoint, centerPoint, pointerPoint, startValue, totalValue):
        """
        get value of pointer meter
        :param startPoint: start point
        :param endPoint: end point
        :param centerPoint: center point
        :param pointerPoint: pointer's outer point
        :param startValue: start value
        :param totalValue: total value
        :return: value
        """
        # print(startPoint, endPoint, centerPoint, pointerPoint, startValue, totalValue)
        angleRange = cls.calAngleClockwise(startPoint, endPoint, centerPoint)
        angle = cls.calAngleClockwise(startPoint, pointerPoint, centerPoint)
        value = angle / angleRange * (totalValue - startValue) + startValue
        if value > totalValue or value < startValue:
            return startValue if angle > np.pi + angleRange / 2 else totalValue
        return value

    @classmethod
    def calPointerValueByPointerVector(cls, startPoint, endPoint, centerPoint, PointerVector, startValue, totalValue):
        """
        get value of pointer meter
        注意传入相对圆心的向量
        :param startPoint: start point
        :param endPoint: end point
        :param centerPoint: center point
        :param PointerVector: pointer's vector
        :param startValue: start value
        :param totalValue: total value
        :return: value
        """
        angleRange = cls.calAngleClockwise(startPoint, endPoint, centerPoint)

        vectorA = startPoint - centerPoint
        vectorB = PointerVector

        angle = cls.__calAngleBetweenTwoVector(vectorA, vectorB)

        # if counter-clockwise
        if np.cross(vectorA, vectorB) < 0:
            angle = 2 * np.pi - angle

        value = angle / angleRange * totalValue + startValue
        if value > totalValue or value < startValue:
            return startValue if angle > np.pi + angleRange / 2 else totalValue
        return value

    def findPointerFromHSVSpace(src, center, radius, radians_low, radians_high, patch_degree=1.0, ptr_resolution=5,
                                low_ptr_color=np.array([0, 0, 221]), up_ptr_color=np.array([180, 30, 255])):

        """
        从固定颜色的区域找指针,未完成
        :param low_ptr_color: 指针的hsv颜色空间的下界
        :param up_ptr_color:  指针的hsv颜色空间的上界
        :param radians_low:圆的搜索范围(弧度制表示)
        :param radians_high:圆的搜索范围(弧度制表示)
        :param src: 二值图
        :param center: 刻度盘的圆心
        :param radius: 圆的半径
        :param patch_degree:搜索梯度，默认每次一度
        :param ptr_resolution: 指针的粗细程度
        :return: 指针遮罩、直线与圆相交的点
        """

    pass


def scanPointer(meter, info):
    """
    find pointer of meter
    :param meter: meter matched template
    :param pts: a list including three numpy array, eg: [startPointer, endPointer, centerPointer]
    :param startVal: an integer of meter start value
    :param endVal: an integer of meter ending value
    :return: pointer reading number
    """
    center = np.array([info["centerPoint"]["x"], info["centerPoint"]["y"]])
    start = np.array([info["startPoint"]["x"], info["startPoint"]["y"]])
    end = np.array([info["endPoint"]["x"], info["endPoint"]["y"]])
    startVal = info["startValue"]
    endVal = info["totalValue"]
    if meter.shape[0] > 500:
        fixHeight = 300
        fixWidth = int(meter.shape[1] / meter.shape[0] * fixHeight)
        resizeCoffX = fixWidth / meter.shape[1]
        meter = cv2.resize(meter, (fixWidth, fixHeight))

        start = (start * resizeCoffX).astype(np.int32)
        end = (end * resizeCoffX).astype(np.int32)
        center = (center * resizeCoffX).astype(np.int32)

    radious = int(EuclideanDistance(start, center))

    src = cv2.GaussianBlur(meter, (3, 3), sigmaX=0, sigmaY=0, borderType=cv2.BORDER_DEFAULT)

    gray = cv2.cvtColor(src=src, code=cv2.COLOR_RGB2GRAY)

    thresh = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, 17, 11)

    mask = np.zeros((src.shape[0], src.shape[1]), np.uint8)
    cv2.circle(mask, (center[0], center[1]), radious, (255, 255, 255), -1)
    thresh = cv2.bitwise_and(thresh, mask)
    cv2.circle(thresh, (center[0], center[1]), int(radious / 3), (0, 0, 0), -1)

    # thresh = cv2.erode(thresh, np.ones((3, 3), np.uint8), 3)
    # thresh = cv2.dilate(thresh, np.ones((5, 5), np.uint8))
    # cv2.imshow("img", thresh)
    # cv2.waitKey(1)

    thresh = cv2.ximgproc.thinning(thresh, thinningType=cv2.ximgproc.THINNING_ZHANGSUEN)

    startAngle = int(
        AngleFactory.calAngleClockwise(startPoint=np.array([center[0] + 100, center[1]]), centerPoint=center,
                                       endPoint=start) * 180 / np.pi)
    endAngle = int(AngleFactory.calAngleClockwise(startPoint=np.array([center[0] + 100, center[1]]), centerPoint=center,
                                                  endPoint=end) * 180 / np.pi)
    # print(startAngle, endAngle)
    if endAngle <= startAngle:
        endAngle += 360
    maxSum = 0
    outerPoint = start
    for angle in range(startAngle - 10, endAngle + 10):
        pts, outPt = getPoints(center, radious, angle)
        thisSum = 0
        showImg = cv2.cvtColor(thresh.copy(), cv2.COLOR_GRAY2BGR)

        for pt in pts:
            cv2.circle(showImg, (pt[0], pt[1]), 2, (0, 0, 255), -1)
            if thresh[pt[1], pt[0]] != 0:
                thisSum += 1

        # cv2.circle(showImg, (outPt[0], outPt[1]), 2, (255, 0, 0), -1)
        # cv2.imshow("img", showImg)
        # cv2.waitKey(1)
        if thisSum > maxSum:
            maxSum = thisSum
            outerPoint = outPt

    if start[0] == outerPoint[0] and start[1] == outerPoint[1]:
        degree = startVal
    elif end[0] == outerPoint[0] and end[1] == outerPoint[1]:
        degree = endVal
    else:
        if start[0] == end[0] and start[1] == end[1]:
            end[0] -= 1
            end[1] -= 3
        degree = AngleFactory.calPointerValueByOuterPoint(start, end, center, outerPoint, startVal, endVal)

    # small value to zero
    if degree - startVal < 0.05 * (endVal - startVal):
        degree = startVal

    if ifShow:
        print(degree, start, center, outerPoint)
        cv2.circle(meter, (outerPoint[0], outerPoint[1]), 10, (0, 0, 255), -1)
        cv2.line(meter, (center[0], center[1]), (outerPoint[0], outerPoint[1]), (0, 0, 255), 5)
        cv2.line(meter, (center[0], center[1]), (start[0], start[1]), (255, 0, 0), 3)
        cv2.line(meter, (center[0], center[1]), (end[0], end[1]), (255, 0, 0), 3)

        thresh = np.expand_dims(thresh, axis=2)
        thresh = np.concatenate((thresh, thresh, thresh), 2)
        meter = np.hstack((meter, thresh))

        cv2.imshow("test", meter)
        cv2.waitKey(0)
    return degree


def EuclideanDistance(pt1, pt2):
    return np.sqrt((pt1[0] - pt2[0]) ** 2 + (pt1[1] - pt2[1]) ** 2)


def getPoints(center, radious, angle):
    res = []
    farthestPointX = int(center[0] + radious * np.cos(angle / 180 * np.pi))
    farthestPointY = int(center[1] + radious * np.sin(angle / 180 * np.pi))

    for ro in range(radious // 3, radious):
        for theta in range(angle - 2, angle + 2):
            angleTemp = theta / 180 * np.pi
            x, y = int(ro * np.cos(angleTemp)) + center[0], int(ro * np.sin(angleTemp)) + center[1]
            res.append([x, y])

    # width = 2
    #
    #
    # for x in range(min(farthestPointX, center[0]), max(farthestPointX, center[0])):
    #     for y in range(min(farthestPointY, center[1]), max(farthestPointY, center[1])):
    #         if k * x + b - 2 <= y <= k * x + b + 2:
    #             res.append([x, y])

    return res, [farthestPointX, farthestPointY]


def findPointerFromBinarySpace(src, center, radius, radians_low, radians_high, patch_degree=1.0, ptr_resolution=5):
    """
    接收一张预处理过的二值图（默认较完整保留了指针信息），从通过圆心水平线右边的点开始，连接圆心顺时针建立直线遮罩，取出遮罩范围下的区域,
    计算对应区域灰度和，灰度和最大的区域即为指针所在的位置。直线遮罩的粗细程度、搜索的梯度决定了算法侦测指针的细粒度。该算法适合搜索指针形状
    为直线的仪表盘，原理与@pointerMaskBySector类似。
    :param radians_low:圆的搜索范围(弧度制表示)
    :param radians_high:圆的搜索范围(弧度制表示)
    :param src: 二值图
    :param center: 刻度盘的圆心
    :param radius: 圆的半径
    :param patch_degree:搜索梯度，默认每次一度
    :param ptr_resolution: 指针的粗细程度
    :return: 指针遮罩、直线与圆相交的点
    """
    _shape = src.shape
    img = src.copy()
    # 弧度转化为角度值
    low = math.degrees(radians_low)
    high = math.degrees(radians_high)
    # _img1 = cv2.erode(_img1, kernel3, iterations=1)
    # _img1 = cv2.dilate(_img1, kernel3, iterations=1)
    # 157=pi/2*100
    mask_info = []
    max_area = 0
    best_theta = 0
    iteration = np.abs(int((high - low) / patch_degree))
    for i in range(iteration):
        # 建立一个大小跟输入一致的全黑图像
        # pointer_mask = np.zeros([_shape[0], _shape[1]], np.uint8)
        # theta = float(i) * 0.01
        # 每次旋转patch_degree度，取圆上一点
        theta = float(i * patch_degree / 180 * np.pi)
        pointer_mask, point = drawLineMask(_shape, theta, center, ptr_resolution, radius)
        # cv2.circle(black_img, (x1, y1), 2, 255, 3)
        # cv2.circle(black_img, (item[0], item[1]), 2, 255, 3)
        # cv2.line(pointer_mask, (center[0], center[1]), point, 255, ptr_resolution)
        # 去除遮罩对应的小区域
        and_img = cv2.bitwise_and(pointer_mask, img)
        not_zero_intensity = cv2.countNonZero(and_img)
        mask_info.append((not_zero_intensity, theta))
        # if not_zero_intensity > mask_intensity:
        #     mask_intensity = not_zero_intensity
        #     mask_theta = theta
        # imwrite(dir_path+'/2_line1.jpg', black_img)
    # 按灰度和从大到小排列
    mask_info = sorted(mask_info, key=lambda m: m[0], reverse=True)
    # thresh = mask_info[0][0] / 30
    # over_index = 1
    # sum = thresh
    # for info in mask_info[1:]:
    #     if mask_info[0][0] - info[0] > thresh:
    #         break
    #     over_index += 1
    best_theta = mask_info[0][1]
    # 得到灰度和最大的那个直线遮罩,和直线与圆相交的点
    pointer_mask, point = drawLineMask(_shape, best_theta, center, ptr_resolution, radius)
    #
    # black_img1 = np.zeros([_shape[0], _shape[1]], np.uint8)
    # r = item[2]-20 if item[2]==_heart[1][2] else _heart[1][2]+ _heart[0][1]-_heart[1][1]-20
    # y1 = int(item[1] - math.sin(mask_theta) * (r))
    # x1 = int(item[0] + math.cos(mask_theta) * (r))
    # cv2.line(black_img1, (item[0], item[1]), (x1, y1), 255, 7)
    # src = cv2.subtract(src, line_mask)
    # img = cv2.subtract(img, line_mask)
    best_theta = 180 - best_theta * 180 / np.pi
    if best_theta < 0:
        best_theta = 360 - best_theta
    return pointer_mask, best_theta, point


def drawLineMask(_shape, theta, center, ptr_resolution, radius):
    """
    画一个长为radius，白色的直线，产生一个背景全黑的白色直线遮罩
    :param _shape:
    :param theta:
    :param center:
    :param ptr_resolution:
    :param radius:
    :return:
    """
    pointer_mask = np.zeros([_shape[0], _shape[1]], np.uint8)
    y1 = int(center[1] - np.sin(theta) * radius)
    x1 = int(center[0] + np.cos(theta) * radius)
    cv2.line(pointer_mask, (center[0], center[1]), (x1, y1), 255, ptr_resolution)
    return pointer_mask, (x1, y1)


def detectHoughLine(meter, cannyThresholds, houghParam):
    '''
    detect pointer of meter
    :param meter:
    :param cannyThresholds: [threshold1, threshold2], parameters of function Canny
    :param houghParam:
    :return:
    '''
    img = cv2.GaussianBlur(meter, (3, 3), 0)  # cv2.imshow("GaussianBlur ", img)
    edges = cv2.Canny(img, cannyThresholds[0], cannyThresholds[1], apertureSize=3)
    # cv2.imshow("canny", edges)
    lines = cv2.HoughLines(edges, 1, np.pi / 180, houghParam)  # 这里对最后一个参数使用了经验型的值
    if lines is None:
        return None
    height, width, _ = img.shape
    pointer = []
    rho, theta = lines[0][0]
    a = np.cos(theta)
    b = np.sin(theta)
    x0 = a * rho
    y0 = b * rho
    # print("x0, y0:", x0, y0)
    # print("width, height ", width, height)
    xcenter = int(width / 2)
    ycenter = int(height / 2)
    if xcenter < x0 or (xcenter == x0 and ycenter > y0):
        x1 = xcenter
        x2 = x0
        y1 = ycenter
        y2 = y0
    else:
        x1 = x0
        x2 = xcenter
        y1 = y0
        y2 = ycenter
    cv2.line(meter, (x1, y1), (x2, y2), (0, 0, 255), 2)
    # cv2.imshow("HoughLine", meter)
    cv2.waitKey(0)
    pointer.append([x2 - x1, y2 - y1])
    pointer = np.array(pointer[0])
    return pointer
