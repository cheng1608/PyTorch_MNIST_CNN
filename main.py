import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import MNIST
import matplotlib.pyplot as plt


def get_data_loader(is_train):
    to_tensor = transforms.Compose([transforms.ToTensor()])
    data_set = MNIST("", is_train, transform=to_tensor, download=True)
    return DataLoader(data_set, batch_size=15, shuffle=True)


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


def evaluate(test_data, net):
    n_correct = 0
    n_total = 0
    with torch.no_grad():
        for (x, y) in test_data:
            outputs = net.forward(x)
            for i, output in enumerate(outputs):
                if torch.argmax(output) == y[i]:
                    n_correct += 1
                n_total += 1
    return n_correct / n_total


def main():
    global x
    train_data = get_data_loader(is_train=True)
    test_data = get_data_loader(is_train=False)
    net = Net()

    print("初始准确率:", evaluate(test_data, net))

    optimizer = torch.optim.Adam(net.parameters(), lr=0.001)

    loss_list = []
    acc_list = []

    for epoch in range(10):
        epoch_loss = 0.0
        for (x, y) in train_data:
            net.zero_grad()
            # print(x.shape)   #torch.Size([15, 1, 28, 28])
            output = net.forward(x)
            loss = torch.nn.functional.nll_loss(output, y)
            loss.backward()  # 反向传播
            optimizer.step()
            epoch_loss += loss.item()

        train_loss = epoch_loss / len(train_data)
        loss_list.append(train_loss)
        accuracy = evaluate(test_data, net)
        acc_list.append(accuracy)
        print("epoch {:d}  准确率: {:.4f}  误差: {:.4f}".format(epoch + 1, accuracy, train_loss))


    epochx = [x for x in range(1, 11)]
    plt.figure()
    plt.subplot(211)
    plt.plot(epochx, loss_list, "r")
    plt.subplot(212)
    plt.plot(epochx, acc_list, "b")
    plt.show()



if __name__ == "__main__":
    main()
