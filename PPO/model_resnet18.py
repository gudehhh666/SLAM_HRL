# import torchvision

# model=torchvision.models.resnet18(pretrained=True)

# print (model)

# depth = np.random.rand(3, 256, 256, 3)



import torch
import torch.nn as nn

from torchvision.models import resnet18 as torchvision_resnet18

import numpy as np

class ResNet18Image(nn.Module):
     def __init__(self):
          super(ResNet18Image, self).__init__()
          resnet_model = torchvision_resnet18(pretrained=True)
          # Remove the last fully connected layer.
          del resnet_model.fc
          self.resnet = resnet_model
     def forward(self, x):
          x = self.resnet.conv1(x)
          x = self.resnet.bn1(x)
          x = self.resnet.relu(x)
          print ('ResNet18Image self.resnet.relu(x): ', x.shape)  #torch.Size([3, 64, 16, 16])
          x = self.resnet.maxpool(x)
          print ('ResNet18Image self.resnet.maxpool(x): ', x.shape)  #torch.Size([3, 64, 8, 8])

          x = self.resnet.layer1(x)
          x = self.resnet.layer2(x)
          x = self.resnet.layer3(x)
          x = self.resnet.layer4(x)
          print ('x.shape: ', x.shape)   #x.shape:  torch.Size([3, 512, 1, 1])

          x = self.resnet.avgpool(x)

          x = x.view(x.size(0), -1)
          # x = self.dropout(x)
          return x




class ResNet18Depth(nn.Module):
     def __init__(self):
          super(ResNet18Depth, self).__init__()
          resnet_model2 = torchvision_resnet18(pretrained=True)
          # Remove the last fully connected layer.
          del resnet_model2.fc
          self.resnet = resnet_model2
     def forward(self, x):
          x = self.resnet.conv1(x)
          x = self.resnet.bn1(x)
          x = self.resnet.relu(x)
          x = self.resnet.maxpool(x)

          x = self.resnet.layer1(x)
          x = self.resnet.layer2(x)
          x = self.resnet.layer3(x)
          x = self.resnet.layer4(x)

          x = self.resnet.avgpool(x)

          x = x.view(x.size(0), -1)
          # x = self.dropout(x)
          return x


class ResNet18Mpt(nn.Module):
     def __init__(self):
          super(ResNet18Mpt, self).__init__()
          resnet_model3 = torchvision_resnet18(pretrained=True)
          # Remove the last fully connected layer.
          del resnet_model3.fc
          self.resnet = resnet_model3
     def forward(self, x):
          x = self.resnet.conv1(x)
          x = self.resnet.bn1(x)
          x = self.resnet.relu(x)
          x = self.resnet.maxpool(x)

          x = self.resnet.layer1(x)
          x = self.resnet.layer2(x)
          x = self.resnet.layer3(x)
          x = self.resnet.layer4(x)

          x = self.resnet.avgpool(x)

          x = x.view(x.size(0), -1)
          # x = self.dropout(x)
          return x


# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# import numpy as np

# class ResidualBlock(nn.Module):
#      def __init__(self, inchannel, outchannel, stride=1):
#           super(ResidualBlock, self).__init__()
#           self.left = nn.Sequential(
#                nn.Conv2d(inchannel, outchannel, kernel_size=3, stride=stride, padding=1, bias=False),
#                nn.BatchNorm2d(outchannel),
#                nn.ReLU(inplace=True),
#                nn.Conv2d(outchannel, outchannel, kernel_size=3, stride=1, padding=1, bias=False),
#                nn.BatchNorm2d(outchannel)
#           )
#           self.shortcut = nn.Sequential()
#           if stride != 1 or inchannel != outchannel:
#                self.shortcut = nn.Sequential(
#                     nn.Conv2d(inchannel, outchannel, kernel_size=1, stride=stride, bias=False),
#                     nn.BatchNorm2d(outchannel)
#                )
               
#      def forward(self, x):
#           out = self.left(x)
#           out = out + self.shortcut(x)
#           out = F.relu(out)
          
#           return out

# class ResNet(nn.Module):
#      def __init__(self, ResidualBlock, num_classes=10):
#           super(ResNet, self).__init__()
#           self.inchannel = 64
#           self.conv1 = nn.Sequential(
#                nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False),
#                nn.BatchNorm2d(64),
#                nn.ReLU()
#           )
#           self.layer1 = self.make_layer(ResidualBlock, 64, 2, stride=1)
#           self.layer2 = self.make_layer(ResidualBlock, 128, 2, stride=2)
#           self.layer3 = self.make_layer(ResidualBlock, 256, 2, stride=2)        
#           self.layer4 = self.make_layer(ResidualBlock, 512, 2, stride=2)        
#           self.fc = nn.Linear(512, num_classes)
          
#      def make_layer(self, block, channels, num_blocks, stride):
#           strides = [stride] + [1] * (num_blocks - 1)
#           layers = []
#           for stride in strides:
#                layers.append(block(self.inchannel, channels, stride))
#                self.inchannel = channels
#           return nn.Sequential(*layers)
     
#      def forward(self, x):
#           out = self.conv1(x)
#           out = self.layer1(out)
#           out = self.layer2(out)
#           out = self.layer3(out)
#           out = self.layer4(out)
#           out = F.avg_pool2d(out, 4)
#           out = out.view(out.size(0), -1)
#           # out = self.fc(out)
#           return out


# def ResNet18():
#     return ResNet(ResidualBlock)

if __name__ == "__main__":
     device0 = torch.device("cuda:0")

     network =  ResNet18Image().to(device0)
     # print (network)
     img = np.random.rand(3, 3, 256, 256)
     img = torch.from_numpy(img).float().to(device0)
     output1 = network(img)
     print (output1.shape)

     net =  ResNet18Depth().to(device0)
     depth = np.random.rand(3, 3, 256, 256)
     depth = torch.from_numpy(depth).float().to(device0)
     output2 = net(depth)
     print (output2.shape)

     nett =  ResNet18Mpt().to(device0)
     mpt = np.random.rand(3, 3, 256, 256)
     mpt = torch.from_numpy(mpt).float().to(device0)
     output3 = nett(depth)
     print (output3.shape)

     cat = torch.cat((output1, output2, output3), 1)
     print (cat.shape)











