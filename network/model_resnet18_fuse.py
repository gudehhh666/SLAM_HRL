import torch
from torch import nn
import numpy as np
import torch.nn.functional as F


# #encoder和decoder的结构
# class ResidualBlock(nn.Module):
#      def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1,
#                     groups=1, bias=False, shortcut=None):
#           super().__init__()
#           if padding == -1:
#                padding = ((np.array(kernel_size) - 1) * np.array(dilation)) // 2
#           self.left = nn.Sequential(
#                nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias),
#                nn.BatchNorm2d(out_channels),
#                nn.ReLU(inplace=True),
#                nn.Conv2d(out_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias),
#                nn.BatchNorm2d(out_channels),
#           )
#           self.right = shortcut
     
#      def forward(self, x):
#           out = self.left(x)
#           residual = x if self.right is None else self.right(x)
#           out += residual
#           return F.relu(out)



# class ResNet18(nn.Module):
#      def __init__(self, in_channel):
#           super().__init__()
#           self.in_channel = in_channel
#           self._build_network()
#      def _build_network(self):
#           # first layer

#           c = 32
#           self.pre = torch.nn.Sequential(
#                torch.nn.Conv2d(self.in_channel, c, kernel_size=3, stride=1, padding=1, bias=False),
#                torch.nn.BatchNorm2d(c),
#                torch.nn.ReLU()
#           )
#           # residual layer * 4 (encoder)
#           self.layer_1 = self._make_layer(c, c, 2)
#           self.layer_2 = self._make_layer2(c, c*2, 2)
#           self.layer_3 = self._make_layer2(c*2, c*4, 2)
#           self.layer_4 = self._make_layer2(c*4, c*8, 2)
          
#           # residual layer * 4 (decoder)
#           self.layer_5 = self._make_layer(c*8, c*4, 2)
#           self.layer_6 = self._make_layer(c*4, c*2, 2)
#           self.layer_7 = self._make_layer(c*2, c, 2)
#           self.layer_8 = self._make_layer(c, 1, 2)
          

#      def _make_layer(self, in_channel, out_channel, block_num):
#           shortcut = torch.nn.Sequential(
#                torch.nn.Conv2d(in_channel, out_channel, 3, 1, 1, bias=False),
#                torch.nn.BatchNorm2d(out_channel)
#           )
#           layers = []
#           layers.append(ResidualBlock(in_channel, out_channel, kernel_size=3, padding=-1, shortcut=shortcut))
#           for _ in range(1, block_num):
#                layers.append(ResidualBlock(out_channel, out_channel, kernel_size=3, padding=-1))
#           return torch.nn.Sequential(*layers)
     
#      def _make_layer2(self, in_channel, out_channel, block_num):
#           shortcut = torch.nn.Sequential(
#                torch.nn.Conv2d(in_channel, out_channel, 3, 2, 1, bias=False),
#                torch.nn.BatchNorm2d(out_channel)
#           )
#           layers = []
#           layers.append(ResidualBlock(in_channel, out_channel, kernel_size=3, stride=2, padding=-1, shortcut=shortcut))
#           for _ in range(1, block_num):
#                layers.append(ResidualBlock(out_channel, out_channel, kernel_size=3, stride=2, padding=-1))
#           return torch.nn.Sequential(*layers)

# #     def forward(self, x):
# #         # first layer
# #         out = self.pre(x)

# #         # encoder
# #         out = self.layer_1(out)
# #         out = self.layer_2(out)
# #         out = self.layer_3(out)
# #         out = self.layer_4(out)
        
# #         # decoder
# #         out = self.layer_5(out)
# #         out = self.layer_6(out)
# #         out = self.layer_7(out)
# #         out = self.layer_8(out)

# #         return out

# class ResNet18SkipConnection(ResNet18):
#      def __init__(self, in_channel):
#           super().__init__(in_channel)

#      def forward(self, x):
#           # first layer
#           out = self.pre(x)

#           # encoder
#           out1 = self.layer_1(out)
#           out2 = self.layer_2(out1)
#           out3 = self.layer_3(out2)
#           out4 = self.layer_4(out3)
#           #    print ('out4: ', out4.shape)
          
#           #    # decoder
#           #    out5 = self.layer_5(out4) + out3
#           #    out6 = self.layer_6(out5) + out2
#           #    out7 = self.layer_7(out6) + out1
#           #    out8 = self.layer_8(out7)

#           return out4


# if __name__ == "__main__":
#      device0 = torch.device("cuda:0")
#      model = ResNet18SkipConnection(in_channel = 5).to(device0)
#      img = np.random.rand(3, 5, 256, 256)
#      img = torch.from_numpy(img).float().to(device0)
#      output1 = model(img)
#      print (output1.shape)



#是三通道的输入，不太好
#参考：https://zhuanlan.zhihu.com/p/157134695?from_voters_page=true
#定义残差块ResBlock
class ResBlock(nn.Module):
    def __init__(self, inchannel, outchannel, stride=1):
        super(ResBlock, self).__init__()
        #这里定义了残差块内连续的2个卷积层
        self.left = nn.Sequential(
            nn.Conv2d(inchannel, outchannel, kernel_size=3, stride=stride, padding=1, bias=False),
            nn.BatchNorm2d(outchannel),
            nn.ReLU(inplace=True),
            nn.Conv2d(outchannel, outchannel, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(outchannel)
        )
        self.shortcut = nn.Sequential()
        if stride != 1 or inchannel != outchannel:
            #shortcut，这里为了跟2个卷积层的结果结构一致，要做处理
            self.shortcut = nn.Sequential(
                nn.Conv2d(inchannel, outchannel, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(outchannel)
            )
            
    def forward(self, x):
        out = self.left(x)
        #将2个卷积层的输出跟处理过的x相加，实现ResNet的基本结构
        out = out + self.shortcut(x)
        out = F.relu(out)
        
        return out



class ResNet(nn.Module):
    def __init__(self, ResBlock, num_classes=10):
        super(ResNet, self).__init__()
        self.inchannel = 64
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False),  #第一个参数代表输入的通道数，如果将RGB、Depth、Mpt合并的话就是5
            nn.BatchNorm2d(64),
            nn.ReLU()
        )
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self.make_layer(ResBlock, 64, 2, stride=1)
        self.layer2 = self.make_layer(ResBlock, 128, 2, stride=2)
        self.layer3 = self.make_layer(ResBlock, 256, 2, stride=2)        
        self.layer4 = self.make_layer(ResBlock, 512, 2, stride=2)        
        self.fc = nn.Linear(512, num_classes)
    #这个函数主要是用来，重复同一个残差块    
    def make_layer(self, block, channels, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.inchannel, channels, stride))
            self.inchannel = channels
        return nn.Sequential(*layers)
    
    def forward(self, x):
        #在这里，整个ResNet18的结构就很清晰了
        out = self.conv1(x)
        # print ('out0.shape: ', out.shape)  #torch.Size([3, 64, 32, 32])
        out = self.maxpool(out)
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        # print ('out1.shape: ', out.shape)  #out1.shape:  torch.Size([3, 512, 1, 1])
        out = F.avg_pool2d(out, 8)
        # print ('out2.shape: ', out.shape)  #out1.shape:  torch.Size([3, 512, 1, 1])
        out = out.view(out.size(0), -1)
        # print ('out3.shape: ', out.shape)  #out3.shape:  torch.Size([3, 512])
        # out = self.fc(out)
        return out


class ResNet_RGB(nn.Module):
    def __init__(self, ResBlock, num_classes=10):
        super(ResNet_RGB, self).__init__()
        self.inchannel = 64
        self.conv1 = nn.Sequential(
            nn.Conv2d(9, 64, kernel_size=7, stride=2, padding=3, bias=False),  #第一个参数代表输入的通道数，如果将RGB、Depth、Mpt合并的话就是5
            nn.BatchNorm2d(64),
            nn.ReLU()
        )
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self.make_layer(ResBlock, 64, 2, stride=1)
        self.layer2 = self.make_layer(ResBlock, 128, 2, stride=2)
        self.layer3 = self.make_layer(ResBlock, 256, 2, stride=2)        
        self.layer4 = self.make_layer(ResBlock, 512, 2, stride=2)        
        self.fc = nn.Linear(512, num_classes)
    #这个函数主要是用来，重复同一个残差块    
    def make_layer(self, block, channels, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.inchannel, channels, stride))
            self.inchannel = channels
        return nn.Sequential(*layers)
    
    def forward(self, x):
        #在这里，整个ResNet18的结构就很清晰了
        out = self.conv1(x)
        # print ('out0.shape: ', out.shape)  #torch.Size([3, 64, 32, 32])
        out = self.maxpool(out)
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        # print ('out1.shape: ', out.shape)  #out1.shape:  torch.Size([3, 512, 1, 1])
        out = F.avg_pool2d(out, 8)
        # print ('out2.shape: ', out.shape)  #out1.shape:  torch.Size([3, 512, 1, 1])
        out = out.view(out.size(0), -1)
        # print ('out3.shape: ', out.shape)  #out3.shape:  torch.Size([3, 512])
        # out = self.fc(out)
        return out




if __name__ == "__main__":
    # #通道合并
    # device0 = torch.device("cuda:0")
    # model = ResNet(ResBlock).to(device0)
    # img = np.random.rand(3, 3, 256, 256)
    # img = torch.from_numpy(img).float().to(device0)
    # depth = np.random.rand(3, 1, 256, 256)
    # depth = torch.from_numpy(depth).float().to(device0)
    # mpt = np.random.rand(3, 1, 256, 256)
    # mpt = torch.from_numpy(mpt).float().to(device0)
    # obv_s = torch.cat((img, depth, mpt), 1)
    # print ('obv_s.shape: ', obv_s.shape)
    # output1 = model(obv_s)
    # print (output1.shape)

    #单独输入
    device0 = torch.device("cuda:0")
    model = ResNet(ResBlock).to(device0)
    img = np.random.rand(5, 3, 320, 240)
    img = torch.from_numpy(img).float().to(device0)
    depth = np.random.rand(5, 3, 320, 240)
    depth = torch.from_numpy(depth).float().to(device0)
    output_img = model(img)
    output_depth = model(depth)
    obv_s = torch.cat((output_img, output_depth), 1)
    print (obv_s.shape)  #torch.Size([5, 1024])


    # mpt = np.random.rand(3, 1, 256, 256)
    # mpt = torch.from_numpy(mpt).float().to(device0)
    # obv_s = torch.cat((img, depth, mpt), 1)
    # print ('obv_s.shape: ', obv_s.shape)
    # output1 = model(obv_s)
    # print (output1.shape)







