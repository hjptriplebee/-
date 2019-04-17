import time

import cv2
import numpy as np

from Algorithm.utils.Finder import meterFinderByTemplate


def isDark(img):
    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    r, c = gray_img.shape[:2]
    # dark num
    dark_sum = 0
    # dark ration
    dark_prop = 0
    # pixels n7m of gray image
    piexs_sum = r * c
    for row in gray_img:
        for colum in row:
            if colum < 120:
                dark_sum += 1
    dark_prop = dark_sum / (piexs_sum)
    # print("dark_sum:" + str(dark_sum))
    # print("piexs_sum:" + str(piexs_sum))
    # print("dark_prop=dark_sum/piexs_sum:" + str(dark_prop))
    if dark_prop >= 0.70:
        return True
    else:
        return False

    # def hist(pic_path):
    #     img = cv2.imread(pic_path, 0)
    #     hist = cv2.calcHist([img], [0], None, [256], [0, 256])
    #     plt.subplot(121)
    #     plt.imshow(img, 'gray')
    #     plt.xticks([])
    #     plt.yticks([])
    #     plt.title("Original")
    #     plt.subplot(122)
    #     plt.hist(img.ravel(), 256, [0, 256])
    #     plt.show()
    #


def readyStatus(img, info):
    template = info['template']
    image = meterFinderByTemplate(img, info['template'])
    # if image is dark enough, do gamma correction for enhancing dark details
    if isDark(image):
        max = np.max(image)
        image = np.power(image / float(max), 1/3) * max
        image = image.astype(np.uint8)
        # cv2.imshow('Gamma', image)
        # cv2.waitKey(0)
    t_shape = template.shape[:2]
    # image = meterFinderBySIFT(img, info)
    orig = image.copy()
    # minimum probability required to inspect a region
    min_confidence = 0.5
    # load the input image and grab the image dimensions
    (H, W) = image.shape[:2]
    # path to input EAST text detector
    model_name = 'frozen_east_text_detection.pb'
    # set the new width and height and then determine the ratio in change
    # for both the width and height
    # image width should be multiple of 32, so do height
    newW = (t_shape[0]) // 32 * 32
    newH = (t_shape[1]) // 32 * 32
    rW = W / float(newW)
    rH = H / float(newH)
    # resize the image and grab the new image dimensions
    image = cv2.resize(image, (newW, newH))
    (H, W) = image.shape[:2]
    # define the two output layer names for the EAST detector model that
    # we are interested -- the first is the output probabilities and the
    # second can be used to derive the bounding box coordinates of text
    layerNames = [
        "feature_fusion/Conv_7/Sigmoid",
        "feature_fusion/concat_3"]
    # load the pre-trained EAST text detector
    # print("[INFO] loading EAST text detector...")
    net = cv2.dnn.readNet(model_name)
    # construct a blob from the image and then perform a forward pass of
    # the model to obtain the two output layer sets
    blob = cv2.dnn.blobFromImage(image, 1.0, (W, H),
                                 (123.68, 116.78, 103.94), swapRB=True, crop=False)
    start = time.time()
    net.setInput(blob)
    (scores, geometry) = net.forward(layerNames)
    end = time.time()
    # show timing information on text prediction
    # print("[INFO] text  detection took {:.6f} seconds".format(end - start))
    # grab the number of rows and columns from the scores volume, then
    # initialize our set of bounding box rectangles and corresponding
    # confidence scores
    (numRows, numCols) = scores.shape[2:4]
    rects = []
    confidences = []
    # loop over the number of rows
    for y in range(0, numRows):
        # extract the scores (probabilities), followed by the geometrical
        # data used to derive potential bounding box coordinates that
        # surround text
        scoresData = scores[0, 0, y]
        xData0 = geometry[0, 0, y]
        xData1 = geometry[0, 1, y]
        xData2 = geometry[0, 2, y]
        xData3 = geometry[0, 3, y]
        anglesData = geometry[0, 4, y]

        # loop over the number of columns
        for x in range(0, numCols):
            # if our score does not have sufficient probability, ignore it
            if scoresData[x] < min_confidence:
                continue

            # compute the offset factor as our resulting feature maps will
            # be 4x smaller than the input image
            (offsetX, offsetY) = (x * 4.0, y * 4.0)

            # extract the rotation angle for the prediction and then
            # compute the sin and cosine
            angle = anglesData[x]
            cos = np.cos(angle)
            sin = np.sin(angle)

            # use the geometry volume to derive the width and height of
            # the bounding box
            h = xData0[x] + xData2[x]
            w = xData1[x] + xData3[x]

            # compute both the starting and ending (x, y)-coordinates for
            # the text prediction bounding box
            endX = int(offsetX + (cos * xData1[x]) + (sin * xData2[x]))
            endY = int(offsetY - (sin * xData1[x]) + (cos * xData2[x]))
            startX = int(endX - w)
            startY = int(endY - h)

            # add the bounding box coordinates and probability score to
            # our respective lists
            rects.append((startX, startY, endX, endY))
            confidences.append(scoresData[x])
    if len(rects) > 0:
        return True
    else:
        return False