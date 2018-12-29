import torch
import torch.nn as nn
import torchvision as tv
import torchvision.transforms as transforms

from torch.nn import functional as F
from torch.autograd import Variable
import torch.optim as optim

from torch.utils.data import TensorDataset, DataLoader

import cv2
import argparse
import numpy as np
import random

device = "cpu"

from model.LeNet import LeNet


parser = argparse.ArgumentParser()
parser.add_argument('--outf', default='./model/', help='folder to output images and model checkpoints')  # 模型保存路径
parser.add_argument('--net', default='./model/net.pth', help="path to netG (to continue training)")  # 模型加载路径
opt = parser.parse_args()

# 超参数设置
EPOCH = 8   # 遍历数据集次数
BATCH_SIZE = 64      # 批处理尺寸(batch_size)
LR = 0.001        # 学习率

# 定义数据预处理方式
transform = transforms.ToTensor()

# 定义训练数据集
trainset = tv.datasets.MNIST(
    root='./data/',
    train=True,
    download=True,
    transform=transform)


# 定义训练批处理数据
trainloader = torch.utils.data.DataLoader(
    trainset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    )

# 定义测试数据集
testset = tv.datasets.MNIST(
    root='./data/',
    train=False,
    download=True,
    transform=transform)

# 定义测试批处理数据
testloader = torch.utils.data.DataLoader(
    testset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    )

# 定义损失函数loss function 和优化方式（采用SGD）
net = LeNet().to(device)
# criterion = nn.CrossEntropyLoss()  # 交叉熵损失函数，通常用于多分类问题上
criterion = nn.NLLLoss()
optimizer = optim.SGD(net.parameters(), lr=LR, momentum=0.9)

# 训练
if __name__ == "__main__":

    for epoch in range(EPOCH):
        sum_loss = 0.0
        # 数据读取
        for i, data in enumerate(trainloader):
            inputs, labels = data

            # 修改minst中的输入，改为二值图
            inputs = torch.ceil(inputs)*255

            inputs, labels = inputs.to(device), labels.to(device)
            # 梯度清零
            optimizer.zero_grad()

            # forward + backward
            outputs = net.forward(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            # 每训练100个batch打印一次平均loss
            sum_loss += loss.item()
            if i % 100 == 99:
                print('epoch %d /step %d: loss:%.03f'
                      % (epoch + 1, i + 1, sum_loss / 100))
                sum_loss = 0.0
        with torch.no_grad():
            correct = 0
            total = 0
            for data in testloader:
                images, labels = data
                images = torch.ceil(images)*255
                images, labels = images.to(device), labels.to(device)
                outputs = net.forward(images)
                # 取得分最高的那个类
                # print(outputs.data)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum()
            print('第%d个epoch的识别准确率为：%d%%' % (epoch + 1, (100 * correct / total)))
        torch.save(net, '%s/net.pkl' % (opt.outf))
