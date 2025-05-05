# 生成模拟验证码数据，使用SEQ2SEQ进行验证码识别，并输出测试数据的识别精度

from PIL import Image,ImageDraw,ImageFont,ImageFilter
import random
import os
import numpy as np

#随机字母
def ranChar():
    # a = str(random.randint(0,9))#数字
    a = chr(random.randint(48,57))#数字
    b = chr(random.randint(65,90))#大写字母
    c = chr(random.randint(97,121))#小写字母
    return random.choice([a,b,c])
    # return random.sample([a,b,c],1)[0]

#随机颜色
def ranColor():
    return (random.randint(32,127),
            random.randint(32, 127),
            random.randint(32, 127))
#240*60
w= 240
h = 60
#创建字体对象
font = ImageFont.truetype("arial.ttf",40)
for i in range(100):
    # 随机生成像素矩阵
    img_arr = np.random.randint(128, 255, (h, w, 3))
    # 将像素矩阵转换成背景图片
    image = Image.fromarray(np.uint8(img_arr))
    # 创建Draw对象
    draw = ImageDraw.Draw(image)
    #填充文字像素颜色
    filename = ""
    for j in range(4):
        ch = ranChar()
        filename+=ch
        # 40+20 是通过增大字体来给字体之间增加间隔
        # draw.text(((40+20)*j+10,10),(ch),font=font,fill=ranColor2())
        # 直接给字体之间增加间隔
        draw.text((40 * j + 20 * (j + 1), 10), (ch), font=font, fill=ranColor())

    # image.show()
    if not os.path.exists("./code"):
        os.makedirs("./code")
    image_path = r"./code"
    image.save("{0}/{1}.jpg".format(image_path,filename))
    print(i)
import os
import torch
import numpy as np
from PIL import Image
import torch.utils.data as data
from torchvision import transforms

data_transforms = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.5,0.5,0.5],std=[0.5,0.5,0.5])
])


class Sampling(data.Dataset):

    def __init__(self,root):
        self.transform = data_transforms
        self.imgs = []
        self.labels = []
        for filenames in os.listdir(root):
            x = os.path.join(root,filenames)
            y = filenames.split('.')[0]
            self.imgs.append(x)
            self.labels.append(y)

    def __len__(self):
        return len(self.imgs)
    def __getitem__(self, index):
        img_path = self.imgs[index]
        img = Image.open(img_path)
        img = self.transform(img)
        label = self.labels[index]
        label = self.one_hot(label)
        return img,label

    def one_hot(self,x):
        z = np.zeros(shape=[4,123])
        for i in range(4):
            index = int(ord(x[i]))
            z[i][index] = 1
        return z

if __name__ == '__main__':

    samping = Sampling("./code")
    dataloader = data.DataLoader(samping,10,shuffle=True,drop_last=True)
    for i,(img_tensor,label) in enumerate(dataloader):
        img = Image.fromarray(np.uint8(((img_tensor[0].data.numpy()*0.5+0.5)*255).transpose(1,2,0)),"RGB")
        # img.show()
        # print(label[0])

        label_1 = np.argmax(label[0],1)
        print(i)
        print(label_1)
        labels = []
        for i in label_1:
            # print(i.item())
            i = chr(i)
            labels.append(i)
        print(labels)
        print(img_tensor.shape)
        print(label.shape)
        # break
import os
import numpy as np
import torch
import torch.nn as nn
import torch.utils.data as data
# from CodeDemo import Sampling_train_chr
from code6_2 import Sampling
from PIL import Image,ImageFont,ImageDraw
import matplotlib.pyplot as plt

class Encoder(nn.Module):
    def __init__(self):
        super(Encoder, self).__init__()
        self.fc1 = nn.Sequential(
            nn.Linear(180,128),
            nn.BatchNorm1d(num_features=128),
            nn.ReLU()
        )
        self.lstm = nn.LSTM(input_size=128,
                            hidden_size=128,
                            num_layers=1,
                            batch_first=True)

    def forward(self, x): # [N,3,60,120]
        # [N,3*60,120]--[N,120,180],//N,V,S--N,S,V
        x = x.reshape(-1,180,240).permute(0,2,1)
        # [N,120,180]--[N*120,180],//N,S,V--N,V
        x = x.reshape(-1,180)
        # [N*120,180]
        fc1 = self.fc1(x)
        # [N*120,128]--# [N,120,128],//N,V--N,S,V
        fc1 = fc1.reshape(-1, 240, 128)
        # [N,120,128],[2,N,128]
        lstm,(h_n,h_c) = self.lstm(fc1,None)
        # [N,128]
        out = lstm[:,-1,:]

        return out


class Decoder(nn.Module):
    def __init__(self):
        super(Decoder, self).__init__()
        self.lstm = nn.LSTM(input_size=128,
                            hidden_size=128,
                            num_layers=1,
                            batch_first=True)
        self.out = nn.Linear(128,123)

    def forward(self,x):
        #[N,128]--[N,1,128],//N,V--N,S,V
        x = x.reshape(-1,1,128)
        #[N,1,128]--[N,4,128],扩维度，广播
        x = x.expand(-1,4,128)
        # [N,4,128],[2,N,128]
        lstm,(h_n,h_c) = self.lstm(x,None)
        # [N,4,128]--[N*4,128],//N,S,V--N,V
        y1 = lstm.reshape(-1,128)
        # [N*4,128]--[N*4,10]
        out = self.out(y1)
        # [N*4,128]--[N,4,122],//N,V--N,S,V
        out = out.reshape(-1,4,123)
        #[N, 4, 122]
        output = torch.softmax(out,dim=2)

        return out,output


class SEQ2SEQ(nn.Module):
    def __init__(self):
        super(SEQ2SEQ, self).__init__()
        self.encoder = Encoder()
        self.decoder = Decoder()

    def forward(self, x):
        encoder = self.encoder(x)
        decoder = self.decoder(encoder)

        return decoder


if __name__ == '__main__':
    BATCH = 64
    EPOCH = 1
    save_dir = r'./params'
    save_path = r'./params/seq.pth'
    font_path = r"./arial.ttf"

    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    seq2seq = SEQ2SEQ().to(device)
    # opt = torch.optim.Adam(seq2seq.encoder.parameters())
    # opt = torch.optim.Adam(seq2seq.decoder.parameters())
    opt = torch.optim.Adam(seq2seq.parameters())
    loss_func = nn.CrossEntropyLoss()

    if os.path.exists(save_path):
        seq2seq.load_state_dict(torch.load(save_path))
    else:
        print("No Params!")

    # train_data = Sampling_train_chr.Sampling(root="./code")
    train_data = Sampling(root="./code")
    train_loader = data.DataLoader(dataset=train_data,
    batch_size=BATCH, shuffle=True, drop_last=True,num_workers=4)

    while True:
        for i, (x, y) in enumerate(train_loader):
            batch_x = x.to(device)
            batch_y = y.float().to(device)

            out,output = seq2seq(batch_x)
            loss = loss_func(out,batch_y)

            opt.zero_grad()
            loss.backward()
            opt.step()

            if i % 10 == 0:
                # [N,4,123]--[N,4]，反one-hot
                label_y = torch.argmax(y,2).data.numpy()
                out_y = torch.argmax(output,2).cpu().data.numpy()
                print(label_y.shape)
                print(out_y.shape)

                #每两组值相等的相加，再除以总数：N*4
                accuracy = np.sum(
                out_y == label_y,dtype=np.float32)/(BATCH * 4)
                print("epoch:{},i:{},loss:{:.4f},acc:{:.2f}%"
                .format(EPOCH,i,loss.item(),accuracy * 100))
                label=[chr(i) for i in label_y[0]]
                out=[chr(i) for i in out_y[0]]
                print("labels:", *label)
                print("outputs:",*out)

                img = (batch_x[0]*0.5+0.5) * 255
                img = img.data.cpu().numpy().transpose(1,2,0)
                image = Image.fromarray(np.uint8(img))
                imgdraw = ImageDraw.ImageDraw(image)
                font = ImageFont.truetype(font_path, size=20)
                imgdraw.text(xy=(0, 0), text=(str(out)), fill="red", font=font)#去掉列表括号
                plt.imshow(image)
                plt.pause(0.1)
        torch.save(seq2seq.state_dict(), save_path)
        EPOCH += 1
        if accuracy > 0.9:
            break

