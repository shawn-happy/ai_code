#  随机生成具有周期性的模拟信号数据，使用1D卷积完成波形信号的拟合回归，并使用测试数据完成测试代码输出R2分数
import torch
#指定N个原始数据做为一个周期的信号基础
seed_data = torch.randint(100,200,(12,),dtype=torch.float32)
#以一定的随机数扰动这个基础周期数据，使得数据具有一定的噪声
# 然后将生成的具有一定噪声的数据组合起来，形成周期数据

#生成训练数据
train_data = []
for i in range(50):
    gen_data = seed_data+torch.randint(-2,2,(len(seed_data),))
    train_data.append(gen_data)
#展开成一维数据
train_data=torch.stack(train_data).reshape(-1)

# 生成测试数据
test_data = []
for i in range(20):
    gen_data = seed_data+torch.randint(-2,2,(len(seed_data),))
    test_data.append(gen_data)
#展开成一维数据
test_data=torch.stack(test_data).reshape(-1)

print(train_data)
print(test_data)
print(train_data.shape)
print(test_data.shape)
torch.save(train_data,"./train.data")
torch.save(test_data,"./test.data")


import torch
import torch.nn as nn
import numpy as np
from sklearn.metrics import r2_score, explained_variance_score
import matplotlib.pyplot as plt

class Net(nn.Module):
    def __init__(self,in_c1,out_c1,out_c2,out_c3,out_c4):
        super().__init__()
        self.conv_1d = nn.Sequential(
            nn.Conv1d(in_c1,out_c1,3,1,0),
            nn.InstanceNorm1d(out_c1),
            nn.ReLU()
        )
        self.conv_2d = nn.Sequential(
            nn.Conv1d(out_c1,out_c2,3,1,0),
            nn.InstanceNorm1d(out_c2),
            nn.ReLU()
        )
        self.conv_3d = nn.Sequential(
            nn.Conv1d(out_c2,out_c3,3,1,0),
            nn.InstanceNorm1d(out_c3),
            nn.ReLU()
        )
        self.fc = nn.Linear(out_c3*3,out_c4)


    def forward(self,x):
        y = self.conv_1d(x)
        y = self.conv_2d(y)
        y = self.conv_3d(y)
        y = y.reshape(y.size(0),-1)
        y = self.fc(y)
        return y

if __name__ == '__main__':

    net = Net(1,256,512,256,1)
    # net.load_state_dict(torch.load("./params.pth"))
    train_data = torch.load("./train.data")
    test_data = torch.load("./test.data")
    print(train_data.shape)
    print(test_data.shape)
    loss_func=nn.MSELoss()
    optim = torch.optim.Adam(net.parameters())
    # exit()
    train = 1
    if train:
        net.train()
        for epoch in range(10):
            for i in range(len(train_data)-9):
                x = train_data[i:i+9]
                y = train_data[i+9:i+10]
                x = x.reshape(-1,1,9)
                y = y.reshape(-1,1)
                out = net(x)
                loss = loss_func(out,y)
                optim.zero_grad()
                loss.backward()
                optim.step()
                print("Epoch:{},Loss:{:.3f}".format(epoch,loss.item()))
            # torch.save(net.state_dict(),"./params.pth")
    net.eval()
    label = []
    output = []
    count = []
    plt.ion()
    for i in range(len(test_data) - 9):
        x = test_data[i:i + 9]
        y = test_data[i + 9:i + 10]
        x = x.reshape(-1, 1, 9)
        y = y.reshape(-1, 1)
        out = net(x)
        loss = loss_func(out, y)
        print(loss.item())
        label.append(y.numpy().reshape(-1))
        output.append(out.data.numpy().reshape(-1))
        count.append(i)
        plt.clf()
        label_icon, = plt.plot(count,label,linewidth=1,color = "blue")
        output_icon, = plt.plot(count,output,linewidth=1, color="red")
        plt.legend([label_icon, output_icon], ["label", "output"], loc="upper right", fontsize=10)

        plt.pause(0.01)
    plt.savefig("./img.pdf")
    plt.ioff()
    plt.show()
    # print(np.shape(label))
    # print(np.shape(output))
    r2=r2_score(label,output)
    variance=explained_variance_score(label,output)
    print(r2)
    print(variance)