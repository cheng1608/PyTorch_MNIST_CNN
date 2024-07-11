>小学期好无聊

老师说要加几个卷积层和池化层，基于 [此版本](https://github.com/cheng1608/Pytorch_MNIST) 改了网络结构

```py
class Net(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(1, 6, 3)
        self.pool1 = torch.nn.MaxPool2d(2)
        self.conv2 = torch.nn.Conv2d(6, 12, 3)
        self.pool2 = torch.nn.MaxPool2d(2)
        self.fc1 = torch.nn.Linear(300, 160)
        self.fc2 = torch.nn.Linear(160, 80)
        self.output4 = torch.nn.Linear(80, 10)

    def forward(self, x):
        x = torch.nn.functional.relu(self.conv1(x))  # [15,6,26,26] (28-1-1)
        x = self.pool1(x)  # [15,6,13,13]
        x = torch.nn.functional.relu(self.conv2(x))  # [15,12,11,11] (13-1-1)
        x = self.pool2(x)  # [15,12,5,5]
        x = x.view(-1, 300)  # [15,300] 12*5*5=300
        x = torch.nn.functional.relu(self.fc1(x))  # [15,160]
        x = torch.nn.functional.relu(self.fc2(x))  # [15,80]
        x = torch.nn.functional.log_softmax(self.output4(x), dim=1)  # [15,10]
        return x
```
