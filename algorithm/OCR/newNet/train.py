from algorithm.OCR.newNet.dataLoader import *
from algorithm.OCR.newNet.LeNet import *

import sys
sys.path.append("../LeNet")

import numpy as np
import torch
import torch.optim as optim
import torch.nn.init as init

torch.manual_seed(10)

bs = 128
lr = 0.001
epoch = 40
stepLength = 20
classes = 11

training = "LCD_enhanced"
data = dataLoader("images/"+training, bs, ifUpdate=True)
testInputs, testLabels = data.getTestData()
print("数据集加载完毕")


weight = np.zeros(classes)

for i in range(classes):
    images = os.listdir("images/LCD_enhanced/train/"+str(i))
    weight[i] = len(images)
weight = weight/np.sum(weight)
print(weight)


def adjust_learning_rate(optimizer, epoch, t=10):
    """Sets the learning rate to the initial LR decayed by 10 every t epochs，default=10"""
    new_lr = lr * (0.1 ** (epoch // t))
    for param_group in optimizer.param_groups:
        param_group['lr'] = new_lr

net = myNet()
net.train()

criterion = nn.NLLLoss(weight=torch.Tensor(weight))
optimizer = optim.Adam(net.parameters(), weight_decay=1e-5, lr=lr)

steps = data.get_rounds()

for e in range(epoch):
    sum_loss = 0.0
    for step in range(steps):

        inputs, labels = data.next_batch()
        inputs = inputs/255
        optimizer.zero_grad()

        # forward + backward
        outputs = net.forward(inputs)
        loss = criterion(outputs, labels.long())
        loss.backward()
        optimizer.step()

        sum_loss += loss.item()
        if step % stepLength == stepLength-1:
            print('epoch %d /step %d: loss:%.03f'
                  % (e + 1, step + 1, sum_loss / stepLength))
            sum_loss = 0.0
    adjust_learning_rate(optimizer, e)
    outputs = net.forward(testInputs)
    _, predicted = torch.max(outputs.data, 1)
    correct = (predicted == testLabels.long()).sum()
    print('第%d个epoch的识别准确率为：%d%%' % (e + 1, (100 * correct / testLabels.shape[0])))
torch.save(net.state_dict(), "net.pkl")

# for i in range(testInputs.shape[0]):
#     test = np.array(testInputs[i].view(28, 28))
#     res = net.forward(testInputs[i].view(1,1,28,28))
#     cv2.imshow("test", test)
#     _, predicted = torch.max(res.data, 1)
#     print(predicted)
#     cv2.waitKey(0)




