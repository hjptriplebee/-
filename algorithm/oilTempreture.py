from algorithm.Common import *
from algorithm.debug import *

red_range = [np.array([156, 120, 80]), np.array([179, 240, 220])]
white_range = [np.array([0, 0, 200]), np.array([180, 30, 255])]


def color_detection(image, color):
    """
    detect red area
    :param image: input image, RGB
    :return: the red area in the image, binary
    """
    HSV = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    Lower = color[0]
    Upper =  color[1]
    mask = cv2.inRange(HSV, Lower, Upper)

    mask = cv2.resize(mask, (0, 0), fx=0.5, fy=0.5)
    kernel = np.ones((3, 3), np.uint8)

    mask = cv2.dilate(mask, kernel,)
    mask = cv2.erode(mask, kernel)
    mask = cv2.blur(mask, (7, 7))
    _, mask = cv2.threshold(mask, 50, 255, cv2.THRESH_BINARY)
    mask = cv2.resize(mask, (0, 0), fx=2, fy=2)

    return mask


def contours_check(image, center):
    """
    find the largest two red area
    find the farthest point to the center point
    :param image:
    :param center:
    :return: the out point
    """
    img, contours, _ = cv2.findContours(image, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
    image = np.expand_dims(image, axis=2)
    image = np.concatenate((image, image, image), axis=-1)
    contours = sorted(contours, key=lambda x: cv2.contourArea(x))
    contours.reverse()
    reach = 0
    target = [0, 0]
    for i in range(len(contours)):
        rec = cv2.minAreaRect(contours[i])
        area = cv2.contourArea(contours[i])
        if area/(rec[1][0]*rec[1][1]+1) < 0.7 or area > 1400 or area < 500:
            continue
        else:
            for point in contours[i]:
                point = np.squeeze(point)
                dis = np.sqrt((point[0]-center[0])**2 + (point[1]-center[1])**2)
                if dis > reach:
                    reach = dis
                    target = point
    return target


def oilTempreture(image, info):
    """
    :param image: ROI image
    :param info: information for this meter
    :return: value
    """
    start = np.array([info["startPoint"]["x"], info["startPoint"]["y"]])
    end = np.array([info["endPoint"]["x"], info["endPoint"]["y"]])
    center = np.array([info["centerPoint"]["x"], info["centerPoint"]["y"]])

    meter = meterFinderBySIFT(image, info["template"])
    h, w, _ = meter.shape

    fixHeight = 300
    fixWidth = int(meter.shape[1] / meter.shape[0] * fixHeight)
    resizeCoffX = fixWidth / meter.shape[1]
    meter = cv2.resize(meter, (fixWidth, fixHeight))

    start = (start * resizeCoffX).astype(np.int16)
    end = (end * resizeCoffX).astype(np.int16)
    center = (center * resizeCoffX).astype(np.int16)

    meter[int(0.6*meter.shape[0]):] *= 0

    mask_meter_red = color_detection(meter, red_range)
    point_red = contours_check(mask_meter_red, center)
    degree_red = AngleFactory.calPointerValueByOuterPoint(start, end, center, point_red, info["startValue"], info["totalValue"])
    cv2.destroyAllWindows()
    mask_meter_white = color_detection(meter, white_range)
    point_white = contours_check(mask_meter_white, center)
    degree_white = AngleFactory.calPointerValueByOuterPoint(start, end, center, point_white, info["startValue"], info["totalValue"])

    if ifShow:
        print("white degree {:.2f}".format(degree_white))
        print("red degree {:.2f}".format(degree_red))
        cv2.circle(meter, (start[0], start[1]), 5, (0, 0, 255), -1)
        cv2.circle(meter, (end[0], end[1]), 5, (0, 0, 255), -1)
        cv2.circle(meter, (center[0], center[1]), 5, (0, 0, 255), -1)
        cv2.circle(meter, (point_white[0], point_white[1]), 5, (255, 255, 255), -1)
        cv2.line(meter, (center[0], center[1]), (point_white[0], point_white[1]), (255,255,255), 5)
        cv2.line(meter, (center[0], center[1]), (point_red[0], point_red[1]), (0, 0, 255), 5)
        cv2.circle(meter, (point_red[0], point_red[1]), 5, (0, 0, 255), -1)
        cv2.imshow("meter", meter)
        cv2.waitKey(0)

    return int(degree_red*100)/100, int(degree_white*100)/100
