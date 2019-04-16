import cv2
import numpy as np
import math
from sklearn.metrics.pairwise import pairwise_distances
from configuration import *

def getPointTemplate(pointID):
    """
    get the locate target
    :param pointID: ID
    :return: the template for location
    """
    return cv2.imread(templatePath + "/"+pointID+".jpg")
    # return cv2.imread("templateForLocation/"+pointID+".jpg")


def locateTargetwithSIFT(image, template):
    """
    get the place of the template in the picture
    :param image: picture captured by robot
    :param template: the saved template
    :return: coordinates x,y,w,h
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

    dst_pts = np.float32([imageKeyPoint[m[0].trainIdx].pt for m in good2]).reshape(-1, 1, 2)
    x, y = int(np.mean(dst_pts[:, 0, 1])), int(np.mean(dst_pts[:, 0, 0]))

    # for show
    # cv2.circle(image, (y, x), 20, (0,0,255), 5)
    # cv2.imshow("show", image)
    # cv2.waitKey(0)
    return [x, y]


def locator(image, pointID):
    """
    global locator
    :param image used for locatomh
    :param pointID: ID of the place
    :return:
    """
    template = getPointTemplate(pointID)
    position = locateTargetwithSIFT(image, template)
    return position


if __name__ == '__main__':
    img = cv2.imread("image/17-2.jpg")
    loc = locator(img, "17-2_1")
    print(loc)