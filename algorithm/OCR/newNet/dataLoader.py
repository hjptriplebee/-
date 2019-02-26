import numpy as np
import torch
import os
import cv2
import pickle
import random


class dataLoader():
    # todo 调整batch，使每个batch顺序都不一样
    def __init__(self, path, bs, ifUpdate):
        self.bs = bs
        self.pointer = 0

        if not os.path.exists(path + "/train.pkl") or not os.path.exists(path + "/test.pkl"):
            ifUpdate = True

        if ifUpdate:
            os.system("rm -rf {}".format(path + "/train.pkl"))
            os.system("rm -rf {}".format(path + "/test.pkl"))
            self.readImagesFromMultiFils(path)

        self.readDataFromPkl(path)
        self.shuffle()

    def readImagesFromMultiFils(self, path):
        for t in ["train", "test"]:
            data = torch.Tensor(np.zeros((1, 1, 28, 28)))
            label = []

            for i in range(11):
                root = path + "/" + t + "/" + str(i) + "/"
                images = os.listdir(root)
                for im in images:
                    if im.split(".")[-1] != "bmp":
                        continue
                    img = cv2.imread(root + im)[:, :, 0]
                    _, thresh = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)
                    temp = torch.Tensor(thresh).view(1, 1, 28, 28)
                    data = torch.cat((data, temp), 0)

                    label.append(i)

            fp = open(path + "/" + t + ".pkl", "wb")
            pickle.dump([data[1:], torch.Tensor(np.array(label)).long()], fp)
            fp.close()

    def readDataFromPkl(self, path):
        with open(path+"/train.pkl", "rb") as fp:
            self.trainData, self.trainLabel = pickle.load(fp)
        with open(path+"/test.pkl", "rb") as fp:
            self.testData, self.testLabel = pickle.load(fp)

    def getTrainData(self):
        return self.trainData, self.trainLabel

    def getTestData(self):
        return self.testData, self.testLabel

    def shuffle(self):
        li = list(range(self.trainData.shape[0]))
        random.shuffle(li)
        self.trainData = self.trainData[li]
        self.trainLabel = self.trainLabel[li]

    def next_batch(self):
        if self.pointer * self.bs == self.trainData.shape[0]:
            self.pointer = 0

        if (self.pointer + 1) * self.bs > self.trainData.shape[0]:
            temp = self.pointer
            self.pointer = 0
            return self.trainData[temp * self.bs:], \
                   self.trainLabel[temp * self.bs:]

        temp = self.pointer
        self.pointer += 1

        return self.trainData[temp * self.bs:self.pointer * self.bs], \
               self.trainLabel[temp * self.bs:self.pointer * self.bs]

    def get_rounds(self):
        return int(self.trainData.shape[0] / self.bs) + 1


if __name__ == "__main__":
    dl = dataLoader("images/LCD_enhanced", 64, True)
    dl.shuffle()
    train, trl = dl.getTrainData()
    test, tel = dl.getTestData()
    #
    print(train.shape, trl)
    print(test.shape, tel)
    #

    print(dl.trainLabel)
    dl.shuffle()
    print(dl.trainLabel)
