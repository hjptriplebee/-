import functools
from algorithm.debug import *
import numpy as np

from algorithm.Common import *
from algorithm.debug import *


def cmp(contour1, contour2):
    return cv2.contourArea(contour2) - cv2.contourArea(contour1)


def cmpCircle(circle1, circle2):
    return circle2[2] - circle1[2]


def checkBleno(image, info):
    """
    :param image: input image
    :param info: image description
    :return:  dumps string by json format ,such as {"meterId": "VALUE"}.
    """
    src = meterFinderByTemplate(image, info['template'])
    src = cv2.resize(src, None, fx=0.2, fy=0.2, interpolation=cv2.INTER_LINEAR)
    blurred = cv2.GaussianBlur(src, (3, 3), 0, None, 0)
    gray = cv2.cvtColor(blurred, cv2.COLOR_RGB2GRAY, None)
    canny = cv2.Canny(gray, 75, 75 * 2, None)
    im, image_contours, hierarchy = cv2.findContours(canny, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    sorted_contours = sorted(image_contours,
                             key=functools.cmp_to_key(cmp))

    # extract the rectangle of ROI
    x_roi, y_roi, w_roi, h_roi = cv2.boundingRect(sorted_contours[0])
    # extract ROI source
    roi = gray[y_roi: y_roi + h_roi, x_roi:x_roi + w_roi]
    # continue search circle  when previous HoughCircle found nothing.
    try_time = 3
    accumulation = 90
    is_find_circles = False
    circles_in_roi = []
    for i in range(0, try_time):
        circles_in_roi = cv2.HoughCircles(roi, cv2.HOUGH_GRADIENT, 2, 10, None, 150, accumulation, 10, 20)
        # adjust the accumulation threshold,find the circle as could as possible
        if circles_in_roi is None:
            accumulation -= 20
            continue
        circles_in_roi = np.uint16(np.round(circles_in_roi))
        circles_in_roi = circles_in_roi[0][:]
        for circle in circles_in_roi:
            # HoughCircles return a [0,0,0] tuple, represents circle not found,
            if circle[2] == 0:
                break
            is_find_circles = True
        accumulation -= 20
        try_time = try_time - 1
    res = {}
    if not is_find_circles:
        res = {"value": -1}
    min_center = min(circles_in_roi, key=lambda c: c[2])
    center_y = min_center[2]
    if info is not None:
        if y_roi + h_roi / 2 > center_y:
            res = {"value": "Up"}
        else:
            res = {"value": "Down"}

    if ifShow:
        cv2.circle(roi, (min_center[0], min_center[1]), min_center[2], (255, 0, 0), cv2.LINE_4)
        cv2.imshow("image", roi)
        cv2.waitKey(0)
    return res



def readBlenometerStatus(image, info):
    print("Blenometer Reader called!!!")
    if image is None:
        print("Resolve Image Error.Inppt image is Empty.")
        return
    return checkBleno(image, info)


# test interface
# if __name__ == '__main__':
# src = cv2.imread('image/IMG_7610.JPG')
# cv2.imshow("Imge",src)
#    res1 = readBlenometerStatus(cv2.imread('image/IMG_7612.JPG'), None)
#    print(res1)
#    info = {"name": "Belometer1"}
#    res2 = readBlenometerStatus(cv2.imread('image/IMG_7612.JPG'), info)
#    print(res2)
