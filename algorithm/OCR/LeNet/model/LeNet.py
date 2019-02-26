import torch.nn as nn


class LeNet(nn.Module):
    def __init__(self):
        super(LeNet, self).__init__()
        self.conv1 = nn.Sequential(     # input_size=(1*28*28)
            nn.Conv2d(1, 6, 5, 1, 2),   # padding=2保证输入输出尺寸相同
            nn.ReLU(),      # input_size=(6*28*28)
            nn.MaxPool2d(kernel_size=2, stride=2)  # output_size=(6*14*14)
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(6, 16, 5),
            nn.ReLU(),      # input_size=(16*10*10)
            nn.MaxPool2d(2, 2)  # output_size=(16*5*5)
        )
        self.fc1 = nn.Sequential(
            nn.Linear(16 * 5 * 5, 120),
            nn.ReLU()
        )
        self.fc2 = nn.Sequential(
            nn.Linear(120, 84),
            nn.ReLU()
        )
        self.fc3 = nn.Linear(84, 10)
        self.softmax = nn.LogSoftmax()

    # 定义前向传播过程，输入为x
    def forward(self, x):
        # print(x)
        x = self.conv1(x)
        # print(x)
        x = self.conv2(x)
        # print(x)
        # nn.Linear()的输入输出都是维度为一的值，所以要把多维度的tensor展平成一维
        x = x.view(x.size()[0], -1)
        x = self.fc1(x)
        # print(x)
        x = self.fc2(x)
        # print(x)
        x = self.fc3(x)
        # print(x)
        # print( self.softmax(x))
        return self.softmax(x)


class myNet(nn.Module):
    def __init__(self):
        super(myNet, self).__init__()
        self.conv1 = nn.Sequential(  # input_size=(1*28*28)
            nn.Conv2d(
                in_channels=1,
                out_channels=16,
                kernel_size=5),  # padding=2保证输入输出尺寸相同
            nn.ReLU(),  # input_size=(6*24*24)
            nn.MaxPool2d(kernel_size=2, stride=2)  # output_size=(6*12*12)
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(
                in_channels=3,
                out_channels=32,
                kernel_size=3),
            nn.ReLU(),  # input_size=(8*10*10)
            nn.MaxPool2d(2, 2)  # output_size=(8*5*5)
        )
        self.fc1 = nn.Sequential(
            nn.Linear(32 * 5 * 5, 128),
            nn.Dropout(0.3),
            nn.ReLU()
        )
        # self.fc2 = nn.Sequential(
        #     nn.Linear(64, 32),
        #     nn.Dropout(0.2),
        #     nn.ReLU()
        # )
        self.fc3 = nn.Linear(32, 10)
        self.softmax = nn.LogSoftmax()

    # 定义前向传播过程，输入为x
    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = x.view(x.size()[0], -1)
        x = self.fc1(x)
        # x = self.fc2(x)
        x = self.fc3(x)
        return self.softmax(x)
