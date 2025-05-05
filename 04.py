# 使用pytorch中的指定方法将训练好的pytorch模型参数文件转换成能够使用以c++语言为主的libtorch框架调用的模型参数文件，以mnist手写数字参数为例进行转换。

import torch
import torch.nn as nn
from torchvision import datasets
import torchvision.transforms as trans
import torch.utils.data as data

class Net(nn.Module):
    def __init__(self):
        super(Net,self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(in_channels=1,out_channels=128,kernel_size=(3,3),stride=1,padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU()
        )#28*28*128

        self.layer2 = nn.Sequential(
            nn.Conv2d(in_channels=128,out_channels=256,kernel_size=(3,3),stride=2,padding=1,groups=4),
            nn.BatchNorm2d(256),
            nn.ReLU()
        )#14*14*256

        self.layer3 = nn.Sequential(
            nn.Conv2d(in_channels=256,out_channels=512,kernel_size=(3,3),stride=2,padding=1,groups=4),
            nn.BatchNorm2d(512),
            nn.ReLU()
        )#7*7*512

        self.layer4 = nn.Sequential(
            nn.Linear(in_features=7*7*512,out_features=10),
        )#10

    def forward(self,x):
        y1 = self.layer1(x)
        y2 = self.layer2(y1)
        y3 = self.layer3(y2)
        y3 = torch.reshape(y3,[x.size(0),-1])
        self.y4 = self.layer4(y3)
        # out = F.softmax(self.y4)
        out = self.y4
        return out

if __name__ == '__main__':
    transf_data = trans.Compose(
        [trans.ToTensor(),
        trans.Normalize(mean=[0.5,],std=[0.5,])]
    )

    train_data = datasets.MNIST("../data",train=True,transform=transf_data,download=True)
    test_data = datasets.MNIST("../data",train=False,transform=transf_data,download=False)
    train_loader = data.DataLoader(train_data,100,shuffle=True)
    test_loader = data.DataLoader(test_data,100,shuffle=True)

    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    net = torch.load("./mnist_net.pth").to(device)# 恢复保存的网络模型
    loss_function = nn.MSELoss()
    # loss_function = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(net.parameters())

    net.train()
    for epoch in range(2):
        for i,(x,y) in enumerate(train_loader):
            x = x.to(device)
            y = y.to(device)
            ys = torch.zeros(y.cpu().size(0), 10).scatter_(1, y.cpu().reshape(-1, 1), 1).cuda()
            output = net(x)
            # loss = loss_function(net.y6,y)
            loss = loss_function(output,ys)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            if i%100 == 0:
                print("Epoch:{},Loss:{:.3f}".format(epoch,loss.item()))

        torch.save(net,"./mnist_net.pth")#保存网络模型


    net.eval()
    eval_loss = 0
    eval_acc = 0
    for i,(x,y) in enumerate(test_loader):
        x = x.to(device)
        y = y.to(device)
        out = net(x)
        ys = torch.zeros(y.cpu().size(0),10).scatter_(1,y.cpu().reshape(-1,1),1).cuda()
        # loss = loss_function(net.y6,ys)
        loss = loss_function(out,ys)
        print("Test_Loss:{:.3f}".format(loss.item()))
        eval_loss += loss.item()*y.size(0)

        arg_max= torch.argmax(out,1)
        eval_acc += (arg_max==y).sum().item()

    mean_loss = eval_loss/len(test_data)
    mean_acc= eval_acc/len(test_data)
    print("Loss:{:.3f},Acc:{:.3f}".format(mean_loss,mean_acc))



import torch
from code10_1 import Net
if __name__ == '__main__':
    device = torch.device('cpu')  # 可以选择使用cpu或cuda进行推理
    model = torch.load("./mnist_net.pth").to(device)  # 加载已有的pytorch模型
    model.eval()  # 把模型转为test模式

    example = torch.randn([1, 1, 28, 28], dtype=torch.float32)
    traced_net = torch.jit.trace(model, example)
    traced_net.save("model.pt")
    torch.onnx.export(model,example,"model.onnx")
    print("模型序列化导出成功")

"""
1、如果模型中有dropout或者batch norm的话，一定要先将模型设置为eval模式，再保存，否则在用libtorch调用后会出现随机干扰；
2、example这个张量的尺寸务必与你自己的模型的输入尺寸一直，否则会出现错误。
3、如果代码中有if条件控制，尽量避免使用torch.jit.trace来转换代码，因为它不能处理变化条件，如果非要用trace的话，可以把if条件控制改成别的形式；
4、jit不能转换第三方Python库中的函数，尽量所有代码都使用pytorch实现，如果速度不理想的话，可以参考github上的pytorch/extension-script项目，用C++实现需要的功能，然后注册成jit操作，最后转成torchscript。

"""