# 在手写数字MNIST数据的训练模型中加入BatchNorm层，和20%-30%的dropout，在训练过程中加入30%的L1正则化项和70%的L2正则化项
import torch
import torch.nn as nn
import torch.utils.data as data
from torch.utils import data
import matplotlib.pyplot as plt
from tensorboardX import SummaryWriter
from torchvision import datasets,transforms
writer = SummaryWriter("./logs")

class Net(nn.Module):
    def __init__(self):
        super(Net,self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(1,64,3,2,1),
            nn.Dropout(0.3),
            nn.BatchNorm2d(64),
            nn.ReLU()
        )
        self.fc1 = nn.Sequential(
            nn.Linear(14*14*64,128),
            nn.Dropout(0.2),
            nn.BatchNorm1d(128),
            nn.ReLU()
        )

        self.fc2 = nn.Linear(128,10)

    def forward(self,x):
        y=self.conv(x)
        y = y.reshape(y.size(0),-1)
        y = self.fc1(y)
        y = self.fc2(y)
        return y

if __name__ == '__main__':
    batch_size=100
    save_params = "./net_params.pth"
    train_data = datasets.MNIST("./data",True,transforms.ToTensor(),download=True)
    test_data = datasets.MNIST("./data",False,transforms.ToTensor(),download=False)
    train_loader = data.DataLoader(train_data,batch_size,shuffle=True)
    test_loader = data.DataLoader(test_data,batch_size,shuffle=True)

    if torch.cuda.is_available():
        device =torch.device("cuda")
    else:
        device = torch.device("cpu")

    net = Net().to(device)
    loss_fn = nn.CrossEntropyLoss()
    optim = torch.optim.Adam(net.parameters(),lr=1e-3, betas=(0.9, 0.999), eps=1e-8,
                 weight_decay=0)

    net.train()
    for epoch in range(1):
        train_loss = 0
        train_acc = 0
        alpha=0.0001
        gamma=0.3
        for i,(x,y) in enumerate(train_loader):
            x = x.to(device)
            y = y.to(device)
            out=net(x)
            L1 = 0
            L2 = 0
            for params in net.parameters():
                L1+=torch.sum(torch.abs(params[0])).to(device)
                L2+=torch.sum(torch.pow(params[0],2)).to(device)
            loss = loss_fn(out,y)
            loss=loss+gamma*alpha*L1+(1-gamma)*alpha*L2
            out = torch.argmax(out, 1)
            train_loss += loss.item()
            train_acc += torch.sum(torch.eq(y.cpu(),out.detach().cpu())).item()
            optim.zero_grad()
            loss.backward(retain_graph=True)
            optim.step()
        train_avgloss = train_loss/len(train_loader)
        train_avgacc = train_acc/len(train_loader)
        print("train_avgloss",train_avgloss)
        print("train_avgacc",train_avgacc)

        test_loss = 0
        test_acc = 0
        for i,(x,y) in enumerate(test_loader):
            x = x.to(device)
            y = y.to(device)
            out = net(x)
            loss = loss_fn(out,y)
            out = torch.argmax(out, 1)
            test_loss+=loss.item()
            test_acc+=torch.sum(torch.eq(y.cpu(),out.detach().cpu())).item()
        test_avgloss = test_loss/len(test_loader)
        test_avgacc = test_acc/len(test_loader)
        print("test_avgloss",test_avgloss)
        print("test_avgacc",test_avgacc)
        # torch.save(net.state_dict(), save_params)



