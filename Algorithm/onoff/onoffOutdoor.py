import json
import numpy as np
import cv2

#type transform
def getMatInt(Mat):

    d = Mat.shape
    for i in range(d[2]):
        for n in range(d[0]):
            for m in range(d[1]):
                Mat[n,m,i] = int(Mat[n,m,i])
                # print(Mat[n,m,i])
    Mat = Mat.astype(np.uint8)
    return Mat
def gamma(image,thre):
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

def backGamma(image,thre):
    """
    :param image: numpy type
    :param thre:float
    :return: image numpy
    """
    f = image / 255.0
    # thre = 0.2 is best for red
    out = np.power(f, thre)

    return out * 255.0
def HSV(img):

    HSV = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    H, S, V = cv2.split(HSV)

    h = H.reshape(H.shape[0]*H.shape[1],order='C')
    s = S.reshape(S.shape[0]*S.shape[1],order='C')
    v = V.reshape(V.shape[0] * V.shape[1], order='C')

    return h,s,v

def GetHsvProperty(h,s,v):

    h_ave = h.mean()
    s_ave = s.mean()
    v_ave = v.mean()
    h_var = h.var()
    s_var = s.var()
    v_var = h.var()

    return h_ave,s_ave,v_ave,h_var,s_var,v_var

def getBlock(image,size=30):
    """
    :param image: 300*300
    :param size:
    :return:
    """
    #the block is 30*30
    # print(image)
    h,w ,_= image.shape
    h_blocks = []

    for i in range(int(h/size)):
        for j in range(int(w/size)):
            img = image[i*size:i*size+size, j*size:j*size + size]
            h,s,v = HSV(img)
            #test
            # print(h)
            h_avg,_,_,_,_,_ = GetHsvProperty(h, s, v)
            h_blocks.append(h_avg)


    # print(img.shape)
    # print(h_blocks)
    # print(len(h_blocks))

    return h_blocks

def countTarPer(h_vec,thre,which):
    green_range_below = 35
    n = 0
    N = 1
    if which =="red":
        for d in h_vec:
            N = N+1
            if d < thre:
                n=n+1
    elif which =="green":
        for d in h_vec:
            N=N+1
            if green_range_below < d < thre:
                n=n+1

    return n,float(n/N)

#use hough get circle
def getCircle(img):

    result = cv2.blur(img, (5, 5))

    gray = cv2.cvtColor(result, cv2.COLOR_BGR2GRAY)

    # canny = cv2.Canny(img, 40, 80)

    circles = cv2.HoughCircles(gray, cv2.HOUGH_GRADIENT, 1, 50, param1=80, param2=30, minRadius=10, maxRadius=40)
    cp_img = None
    # print(circles)
    #if circles != None:
    for circle in circles[0]:
        # print(circle[2])

        x = int(circle[0])
        y = int(circle[1])

        # r = int(circle[2])
        # print("r========" + str(r))

        # img = cv2.circle(img, (x, y), r, (0, 0, 255), 1, 8, 0)

        cp_img = img[y - 10:y + 30, x - 10:x + 30]

    # cv2.imwrite("cut.jpg", cp_img)

    return cp_img

def onoffOutdoor(image, info):
    color = ""
    num = -1
    per = -1
    switch_thre = info["switchThreshold"]
    red_range_above = info["redRangeAbove"]
    green_range_above = info["greenRangeAbove"]
    red_num_thre = info["redNumThreshold"]
    green_num_thre = info["greenNumThreshold"]
    # the second par need to be altered according to conditions,such as red or blue
    image = gamma(image, switch_thre)
    # the step need

    # use hough to locate the circle
    # to compatible yaxi test images ,we annotate the way to search circle by hough
    # the test result is good
    #image = getCircle(image)

    # cv2.imwrite("temp.jpg",image)
    vectors = getBlock(image,5)

    # print("----vectors")
    # print(vectors)
    #find red  range 0-40
    red_num,red_per = countTarPer(vectors, red_range_above,"red")

    # the step of red is prior
    if red_num>red_num_thre:
        color = "red"
        num = red_num
        per = red_per
    else:
        # find blue range
        green_num, green_per = countTarPer(vectors, green_range_above, "green")
        if green_num>=green_num_thre:
            color = "green"
            num = green_num
            per = green_per

    res = {
        'color': color,
        'num':num,
        'per':per
    }
    res = json.dumps(res)

    return res
