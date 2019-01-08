from Common import *
import json
import util.PlotUtil as plot

plot_index = 0


def normalPressure(image, info):
    import showLabel as sl
    # sl.showLabel(image, info)
    return readPressure(image, info)


def inc():
    global plot_index
    plot_index += 1
    return plot_index


def reset():
    global plot_index
    plot_index = 0


def readPressure(image, info):
    src = meterFinderByTemplate(image, info["template"])
    pyramid = 0.5
    plot.subImage(src=src, index=inc(), title='Template Src')
    if 'pyramid' in info and info['pyramid'] is not None:
        pyramid = info['pyramid']
        src = cv2.resize(src, (0, 0), fx=pyramid, fy=pyramid)
    src = cv2.GaussianBlur(src, (3, 3), sigmaX=0, sigmaY=0, borderType=cv2.BORDER_DEFAULT)
    gray = cv2.cvtColor(src=src, code=cv2.COLOR_RGB2GRAY)
    thresh = gray.copy()
    cv2.threshold(gray, 120, 255, cv2.THRESH_BINARY_INV, thresh)
    thresh = cv2.ximgproc.thinning(thresh, thinningType=cv2.ximgproc.THINNING_ZHANGSUEN)
    # plot.subImage(src=thresh, index=inc(), title='thresh', cmap='gray')
    do_hist = info["enableEqualizeHistogram"]
    if do_hist:
        gray = cv2.equalizeHist(gray)
    canny = cv2.Canny(src, 75, 75 * 2)
    dilate_kernel = cv2.getStructuringElement(ksize=(3, 3), shape=cv2.MORPH_ELLIPSE)
    erode_kernel = cv2.getStructuringElement(ksize=(1, 1), shape=cv2.MORPH_ELLIPSE)
    # fill scale line with white pixels
    canny = cv2.dilate(canny, dilate_kernel)
    canny = cv2.erode(canny, erode_kernel)
    # find contours
    img, contours, hierarchy = cv2.findContours(thresh, mode=cv2.RETR_LIST, method=cv2.CHAIN_APPROX_NONE)
    # filter the large contours, the pixel number of scale line should be small enough.
    # and the algorithm will find the pixel belong to the scale line as we need.
    contours_thresh = info["contoursThreshold"]
    contours = [c for c in contours if len(c) > contours_thresh]
    # draw contours
    src = cv2.cvtColor(src, cv2.COLOR_BGR2RGB)
    filtered_thresh = np.zeros(thresh.shape, dtype=np.uint8)
    cv2.drawContours(filtered_thresh, contours, -1, (255, 0, 0), thickness=cv2.FILLED)
    thresh = filtered_thresh
    plot.subImage(src=filtered_thresh, index=inc(), title='Filtered Threshold', cmap='gray')
    # prasan_iteration = rasan.getIteration(0.7, 0.3)
    # load meter calibration form configuration
    double_range = info['doubleRange']
    start_ptr = info['startPoint']
    end_ptr = info['endPoint']
    ptr_resolution = info['ptrResolution']
    clean_ration = info['cleanRation']
    start_ptr = cvtPtrDic2D(start_ptr)
    end_ptr = cvtPtrDic2D(end_ptr)
    center = 0  # 表盘的中心
    radius = 0  # 表盘的半径
    center = info['centerPoint']
    center = cvtPtrDic2D(center)
    # 起点和始点连接，分别求一次半径,并得到平均值
    radius = calAvgRadius(center, end_ptr, radius, start_ptr)
    # 清楚可被清除的噪声区域，噪声区域(文字、刻度数字、商标等)的area 可能与指针区域的area形似,应该被清除，
    # 防止在识别指针时出现干扰。值得注意，如果当前指针覆盖了该干扰区域，指针的一部分可能也会被清除
    # 用直线Mask求指针区域
    hlt = np.array([center[0] - radius, center[1]])  # 通过圆心的水平线与圆的左交点
    # 计算起点向量、终点向量与过圆心的左水平线的夹角
    start_radians = AngleFactory.calAngleClockwise(start_ptr, hlt, center)
    # 以过圆心的左水平线为扫描起点
    if start_radians < np.pi:
        # 在水平线以下,标记为负角
        start_radians = -start_radians
    end_radians = AngleFactory.calAngleClockwise(hlt, end_ptr, center)
    # 从特定范围搜索指针
    pointer_mask, theta, line_ptr, cm = findPointerFromBinarySpace(thresh, center, radius, start_radians,
                                                                   end_radians,
                                                                   patch_degree=0.5,
                                                                   ptr_resolution=ptr_resolution)
    line_ptr = cv2PtrTuple2D(line_ptr)
    # plot.subImage(src=canny, index=inc(), title='canny', cmap='gray')
    plot.subImage(src=cv2.bitwise_or(thresh, pointer_mask), index=inc(), title='pointer', cmap='gray')
    plot.subImage(src=cm, index=inc(), title='cleaned', cmap='gray')
    cv2.line(src, (start_ptr[0], start_ptr[1]), (center[0], center[1]), color=(0, 0, 255), thickness=1)
    cv2.line(src, (end_ptr[0], end_ptr[1]), (center[0], center[1]), color=(0, 0, 255), thickness=1)
    cv2.circle(src, (start_ptr[0], start_ptr[1]), 5, (0, 0, 255), -1)
    cv2.circle(src, (end_ptr[0], end_ptr[1]), 5, (0, 0, 255), -1)
    cv2.circle(src, (center[0], center[1]), 2, (0, 0, 255), -1)
    plot.subImage(src=cv2.cvtColor(src, cv2.COLOR_BGR2RGB), index=inc(), title='calibration')
    if double_range:
        start_value_in = info['startValueIn']
        total_value_in = info['totalValueIn']
        start_value_out = info['startValueOut']
        total_value_out = info['totalValueOut']
        valueIn = AngleFactory.calPointerValueByPoint(startPoint=start_ptr, endPoint=end_ptr, centerPoint=center,
                                                      point=line_ptr, startValue=start_value_in,
                                                      totalValue=total_value_in)
        valueOut = AngleFactory.calPointerValueByPoint(startPoint=start_ptr, endPoint=end_ptr, centerPoint=center,
                                                       point=line_ptr, startValue=start_value_out,
                                                       totalValue=total_value_out)
        return json.dumps({
            "valueIn": valueIn,
            "valueOut": valueOut
        })
    else:
        start_value = info['startValue']
        total = info['totalValue']
        value = AngleFactory.calPointerValueByPoint(startPoint=start_ptr, endPoint=end_ptr,
                                                    centerPoint=center,
                                                    point=line_ptr, startValue=start_value,
                                                    totalValue=total)
        return json.dumps({"value": value})


def calAvgRadius(center, end_ptr, radius, start_ptr):
    radius_1 = np.sqrt(np.power(start_ptr[0] - center[0], 2) + np.power(start_ptr[1] - center[1], 2))
    radius_2 = np.sqrt(np.power(end_ptr[0] - center[0], 2) + np.power(end_ptr[1] - center[1], 2))
    radius = np.int64((radius_1 + radius_2) / 2)
    return radius


def cvtPtrDic2D(dic_ptr):
    """
    point.x,point.y转numpy数组
    :param dic_ptr:
    :return:
    """
    if dic_ptr['x'] and dic_ptr['y'] is not None:
        dic_ptr = np.array([dic_ptr['x'], dic_ptr['y']])
    else:
        return np.array([0, 0])
    return dic_ptr


def cv2PtrTuple2D(tuple):
    """
    tuple 转numpy 数组
    :param tuple:
    :return:
    """
    if tuple[0] and tuple[1] is not None:
        tuple = np.array([tuple[0], tuple[1]])
    else:
        return np.array([0, 0])
    return tuple




def cv2PtrTuple2D(tuple):
    """
    tuple 转numpy 数组
    :param tuple:
    :return:
    """
    if tuple[0] and tuple[1] is not None:
        tuple = np.array([tuple[0], tuple[1]])
    else:
        return np.array([0, 0])
    return tuple


def readPressureValueFromDir(meter_id, img_dir, config):
    img = cv2.imread(img_dir)
    file = open(config)
    info = json.load(file)
    assert info is not None
    info["template"] = cv2.imread("template/" + meter_id + ".jpg")
    return readPressureValueFromImg(img, info)


def readPressureValueFromImg(img, info):
    if img is None:
        raise Exception("Failed to resolve the empty image.")
    return normalPressure(img, info)


if __name__ == '__main__':
    res1 = readPressureValueFromDir('lxd1_4', 'image/lxd1.jpg', 'config/lxd1_4.json')
    res2 = readPressureValueFromDir('szk2_1', 'image/szk2.jpg', 'config/szk2_1.json')
    res3 = readPressureValueFromDir('szk1_5', 'image/szk1.jpg', 'config/szk1_5.json')
    res4 = readPressureValueFromDir('wn1_5', 'image/wn1.jpg', 'config/wn1_5.json')
    res5 = readPressureValueFromDir('xyy3_1', 'image/xyy3.jpg', 'config/xyy3_1.json')
    res6 = readPressureValueFromDir('pressure2_1', 'image/pressure2.jpg', 'config/pressure2_1.json')
    print(res1)
    print(res2)
    print(res3)
    print(res4)
    print(res5)
    print(res6)
