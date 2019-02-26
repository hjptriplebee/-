import sys
import cv2
import os
import numpy as np
import torch
import random

from collections import defaultdict
import tensorflow as tf
from keras.models import load_model
from keras import backend as K
from algorithm.debug import *

sys.path.append(".")


def fillAndResize(image):
    """
    将输入图像填充为正方形且变换为（28，28）
    :param image:
    :return:
    """
    h, w = image.shape
    l = max(w, h)
    ret = np.zeros((l, l), np.uint8)
    leftTop = np.array([l/2-w/2, l/2-h/2], np.uint8)
    ret[leftTop[1]:leftTop[1]+h, leftTop[0]:leftTop[0]+w] = image
    ret = cv2.resize(ret, (28, 28), interpolation=cv2.INTER_CUBIC)
    return ret


class leNetOCR:
    def __init__(self):
        """
        初始化LeNet模型
        :return:
        """
        sys.path.append("LeNet")
        self.net = torch.load("algorithm/OCR/LeNet/model/net.pkl")

    def recognizeNet(self, image):
        """
        LeNet识别图像中的数字
        :param image: 输入图像
        :return: 识别的数字值
        """
        image = fillAndResize(image)
        if ifShow:
            cv2.imshow("single", image)
            cv2.waitKey(0)
        if image.size != 784:
            print("检查输入图片大小！不为28*28")
            return None
        image = torch.Tensor(image).view((1, 1, 28, 28))
        image = image.to("cpu")
        result = self.net.forward(image)
        _, predicted = torch.max(result.data, 1)
        num = int(np.array(predicted[0]).astype(np.uint32))
        return num

class Cnn(object):
    def __init__(self):
        sys.path.append("tfNet")
        self.model = load_model('algorithm/OCR/tfNet/CNN.h5')

    #将图像等比例变为28*28
    def resize_28(self,number):
        mask = np.zeros((28, 28))
        number_row = (28 - number.shape[0]) // 2
        number_col = (28 - number.shape[1]) // 2
        for i in range(number.shape[0]):
            for j in range(number.shape[1]):
                mask[i + number_row][j + number_col] = number[i][j]
        number = mask
        return number

    #数字识别
    def recognizeNet(self,image):
        """
        :param image:  确保是“黑底白字”的二值图，（因为二值化后图可能是“白底黑字”）
        :return: 返回字符型的数字  其中n代表不是数字
        """
        #将图像等比例变为28*28
        h, w = image.shape
        if h > 28 or w > 28:
            if h >= w:
                scale = round(26 / h, 2)
                image = cv2.resize(image, (0, 0), fx=scale, fy=scale)
                image = self.resize_28(image)
            else:
                scale = round(26 / w, 2)
                image = cv2.resize(image, (0, 0), fx=scale, fy=scale)
                image = self.resize_28(image)
        else:
            image =self. resize_28(image)
        # 确保是0和255
        image = np.array([[0 if y < 150 else 255 for y in x] for x in image])

        image = np.expand_dims(image, axis=0)
        image = np.expand_dims(image, axis=3)
        result = self.model.predict(image)
        maxProb = -float('inf')
        num = ""
        for i in range(len(result[0])):
            if maxProb < result[0][i]:
                maxProb = result[0][i]
                if i == 10:
                    num = 'n'
                else:
                    num = str(i)
        return num

class tfNet(object):
    def __init__(self):
        """
        初始化模型
        """
        sys.path.append("tfNet")
        self.sess = tf.Session()
        self.saver = tf.train.import_meta_graph('algorithm/OCR/tfNet/model.meta')
        self.saver.restore(self.sess, tf.train.latest_checkpoint('algorithm/OCR/tfNet'))
        self.graph = tf.get_default_graph()

    def recognizeNet(self, test):
        test = fillAndResize(test)
        new_img = cv2.resize(test, (28, 28))
        new_img = new_img.reshape(-1, 784)
        new_img = np.minimum(new_img, 1)
        image = self.graph.get_tensor_by_name("img:0")
        predict = self.graph.get_tensor_by_name("predict:0")
        prediction = self.sess.run(predict, feed_dict={image: new_img})
        print("predict number:", prediction[0])
        return prediction[0]

    def recognizeNet2(self, test):
        # test = fillAndResize(test)
        # new_img = cv2.resize(test, (28, 28))
        # new_img = new_img.reshape(-1, 784)
        # new_img = np.minimum(new_img, 1)
        image = self.graph.get_tensor_by_name("img:0")
        predict = self.graph.get_tensor_by_name("predict:0")
        prediction = self.sess.run(predict, feed_dict={image: test})
        # print("predict number:", prediction[0])
        return prediction[0]


def zero():
    return 0


class newNet(object):

    def __init__(self):
        """
        初始化LeNet模型
        :return:
        """
        sys.path.append("newNet")
        from algorithm.OCR.newNet.LeNet import myNet

        self.net = myNet()
        self.net.eval()
        self.net.load_state_dict(torch.load("algorithm/OCR/newNet/net.pkl"))

    def recognizeNet(self, image):
        """
        LeNet识别图像中的数字
        :param image: 输入图像
        :return: 识别的数字值
        """
        image = fillAndResize(image)
        tensor = torch.Tensor(image).view((1, 1, 28, 28))/255

        tensor = tensor.to("cpu")
        result = self.net.forward(tensor)
        _, predicted = torch.max(result.data, 1)
        num = int(np.array(predicted[0]).astype(np.uint32))

        if not os.path.exists("storeDigitData"):
            os.system("mkdir storeDigitData")
        imgNum = len(os.listdir("storeDigitData/"))
        cv2.imwrite("storeDigitData/" + str(imgNum) + "_" + str(num) + ".bmp", image)

        if ifShow:
            print(num)
            cv2.imshow("single", image)
            cv2.waitKey(0)

        return str(num) if num != 10 else "?"
