import json
import numpy as np
import cv2
from algorithm.Common import AngleFactory, meterFinderByTemplate


# type transform


def getMatInt(Mat):
    d = Mat.shape
    for i in range(d[2]):
        for n in range(d[0]):
            for m in range(d[1]):
                Mat[n, m, i] = int(Mat[n, m, i])

    Mat = Mat.astype(np.uint8)
    return Mat


def gamma(image, thre):
    """
    :param image: numpy type
    :param thre:float
    :return: image numpy
    """

    f = image / 255.0
    # we can change thre accoding  to real condition
    # thre = 0.3
    out = np.power(f, thre)
    out = getMatInt(out * 255)

    return out


def backGamma(image, thre):
    """
    :param image: numpy type
    :param thre:float
    :return: image numpy
    """
    f = image / 255.0
    out = np.power(f, thre)
    return out * 255.0


def HSV(img):
    HSV = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    H, S, V = cv2.split(HSV)
    h = H.reshape(H.shape[0] * H.shape[1], order='C')
    s = S.reshape(S.shape[0] * S.shape[1], order='C')
    v = V.reshape(V.shape[0] * V.shape[1], order='C')
    return h, s, v


def GetHsvProperty(h, s, v):
    h_ave = h.mean()
    s_ave = s.mean()
    v_ave = v.mean()
    h_var = h.var()
    s_var = s.var()
    v_var = h.var()
    return h_ave, s_ave, v_ave, h_var, s_var, v_var


# 针对红颜色吸湿器,备用
def RedReader(image, info):
    """
    :param image: ROI image
    :param info: information for this meter
    :return: {
       color:
       h_ave:
       s_ave:
       v_ave:
       h_var:
       s_var:
       v_var:
    }
    """
    # Gamma transformation of original image
    image = gamma(image, 0.2)

    # the step need
    cv2.imwrite("gamma.jpg", image)
    image = cv2.imread("gamma.jpg")

    meter = (image, info["template"])

    h, s, v = HSV(meter)
    h_ave, s_ave, v_ave, h_var, s_var, v_var = GetHsvProperty(h, s, v)
    cv2.imwrite("meter.jpg", meter)

    res = {
        'color': -1,
        'h_ave': h_ave,
        's_ave': s_ave,
        'v_ave': v_ave,
        'h_var': h_var,
        's_var': s_var,
        'v_var': v_var
    }
    res = json.dumps(res)
    return res


def getBlock(image, size=30):
    """
    :param image: 300*300
    :param size:
    :return:
    """
    # the block is 30*30
    h, w, _ = image.shape
    h_blocks = []

    for i in range(int(h / size)):
        for j in range(int(w / size)):
            img = image[i * size:i * size + size, j * size:j * size + size]
            h, s, v = HSV(img)
            # test
            h_avg, _, _, _, _, _ = GetHsvProperty(h, s, v)
            h_blocks.append(h_avg)
    return h_blocks


def countTarPer(h_vec, thre, which):
    blue_below_thre = 100
    n = 0
    N = 0
    if which == "red":
        for d in h_vec:
            N = N + 1
            if d < thre:
                n = n + 1
    elif which == "blue":
        for d in h_vec:
            N = N + 1
            if blue_below_thre < d < thre:
                n = n + 1
    return n, float(n / N)


def absorb(image, info):
    # vars init
    color = ""
    num = -1
    per = -1
    red_range_thre = info["redRangeThreshold"]
    blue_range_thre = info["blueRangeThreshold"]
    red_num_thre = info["redNumThreshold"]
    blue_num_thre = info["blueNumThre"]
    # the second par need to be altered according to conditions,such as red or blue
    image = gamma(image, 0.4)
    vectors = getBlock(image)
    # find red  range 0-40
    red_num, red_per = countTarPer(vectors, red_range_thre, "red")

    # the step of red is prior
    if red_num >= red_num_thre:
        color = "red"
        num = red_num
        per = red_per
    else:
        # find blue range
        blue_num, blue_per = countTarPer(vectors, blue_range_thre, "blue")

        if blue_num >= blue_num_thre:
            color = "blue"
            num = blue_num
            per = blue_per

    res = {
        'color': color,
        'num': num,
        'per': per
    }

    res = json.dumps(res)

    return res
