# 使用2D卷积与池化和全连接结合的模型实现CIFAR10图像的识别，同时实现测试部分代码,输出测试数据的平均精度和损失
import torch.nn as nn
import torch
import torch.utils.data as data
from torchvision import datasets,transforms
class Net(nn.Module):
    def __init__(self):
        super().__init__()

        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=16, kernel_size=(3,3), stride=(1,1),padding=1),#N,16,32,32
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(2,2))#N,16,16,16

        self.conv2 = nn.Sequential(
            nn.Conv2d(in_channels=16, out_channels=64, kernel_size=(3,3), stride=(1,1),padding=1,groups=8),#N,64,16,16
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d((2,2),2))#N,64,8,8

        self.fc1 = nn.Sequential(
            nn.Linear(in_features=64*8*8,out_features=128),
            nn.BatchNorm1d(128),
            nn.ReLU())#N,128

        self.fc2 = nn.Linear(in_features=128,out_features=10)#N,10

    def forward(self,x):
        y=self.conv1(x)
        y=self.conv2(y)
        y=y.reshape(y.size(0),-1)
        y=self.fc1(y)
        y=self.fc2(y)
        return y

if __name__ == '__main__':
    batch_size=100
    save_params = "./net_params.pth"
    save_net = "./net.pth"
    train_data = datasets.CIFAR10("./data",True,transforms.ToTensor(),download=True)
    test_data = datasets.CIFAR10("./data",False,transforms.ToTensor(),download=False)
    train_loader = data.DataLoader(train_data,batch_size,shuffle=True)
    test_loader = data.DataLoader(test_data,batch_size,shuffle=True)

    if torch.cuda.is_available():
        device =torch.device("cuda")
    else:
        device = torch.device("cpu")

    net = Net().to(device)
    # net.load_state_dict(torch.load(save_params))
    loss_fn = nn.CrossEntropyLoss()#自动对网络输出做softmax缩放，自动对标签求one-hot
    optim = torch.optim.Adam(net.parameters())

    net.train()
    for epoch in range(1):
        for i,(x,y) in enumerate(train_loader):
            x = x.to(device)
            y = y.to(device)
            out=net(x)
            loss = loss_fn(out,y)
            optim.zero_grad()
            loss.backward()
            optim.step()
            if i%100 == 0:
                print("epoch:{},loss:{:.3f}".format(epoch,loss))
            torch.save(net.state_dict(),save_params)
            torch.save(net,save_net)

    net.eval()
    eval_loss = 0
    eval_acc = 0
    for i,(x,y) in enumerate(test_loader):
        x = x.to(device)
        y = y.to(device)
        out = net(x)
        loss = loss_fn(out,y)
        eval_loss+=loss.item()*batch_size
        max_out = torch.argmax(out,1)
        eval_acc += (max_out==y).sum().item()
    mean_loss = eval_loss/len(test_data)
    mean_acc = eval_acc/len(test_data)
    print(mean_loss,mean_acc)