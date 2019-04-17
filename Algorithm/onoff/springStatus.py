from Algorithm.utils.Finder import meterFinderByTemplate


# 识别弹簧状态
def springStatus(ROI, info):
    template = info['template']
    img = meterFinderByTemplate(ROI, template)
    high, width = img.shape[:2]
    i = 0
    y = 2 * high // 5  # y坐标大约是2/5的高度
    while width - 6 - i > 0:  # 比较相邻像素的三个通道的差值   从右往左比较
        a1, a2 = img[y][width - 5 - i][0].astype(int), img[y][width - 5 - i - 1][0].astype(
            int)  # 这里的5是为了防止扣取的模板不精确（防止边框抠出来的情况）
        b1, b2 = img[y][width - 5 - i][1].astype(int), img[y][width - 5 - i - 1][1].astype(int)
        c1, c2 = img[y][width - 5 - i][2].astype(int), img[y][width - 5 - i - 1][2].astype(int)
        if abs(a1 - a2) < 20 and abs(b1 - b2) < 20 and abs(c1 - c2) < 20:  # 如果相邻像素值近似，继续循环
            i += 1
        else:  # 不想死，输出当前点的x值
            x = width - 5 - i - 1
            break
    # cv2.circle(img, (x, y), 2, (0, 0, 255), -1)
    # cv2.imshow("ewe", img)
    if x > 2 * width // 3:
        return {'value': '100%'}
    else:
        return {'value': '50%'}
