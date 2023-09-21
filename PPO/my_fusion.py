import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F

device0 = torch.device("cuda:0")
from thop import profile
from torchsummary import summary

class Fusion_rgbdepth(nn.Module):
     def __init__(self, bias=True):
          super(Fusion_rgbdepth, self).__init__()
          self.gate_rear1 = nn.Conv2d(128, 64, 1, padding=0, bias=False)
          self.bn1 = nn.BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True)

          self.relu1_2 = nn.ReLU(inplace=True)
          self.sconv1_2 = nn.Conv2d(64, 64, 1, padding=0, bias=False)
          self.sconv_bn1 = nn.BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True)
          self.sconv_relu1 = nn.ReLU(inplace=True)
          self.cat_feature_out1 = nn.Sequential(nn.Conv2d(64, 16, 3, padding=1), nn.ReLU(inplace=True),
                                              nn.Conv2d(16, 1, 1, padding=0)
                                             )
          self.relu_RU5_img = nn.PReLU()
          self.conv_RU5_1_img = nn.Conv2d(64, 64, 3, padding=1, bias=False)
          self.bn_RU5_1_img = nn.BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True)
          self.relu_RU5_1_img = nn.PReLU()
          self.conv_RU5_2_img = nn.Conv2d(64, 64, 3, padding=1, bias=False)
          self.bn_RU5_2_img = nn.BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True)

          self.relu_RU_end5_img = nn.PReLU()

          self.relu_RU5_depth = nn.PReLU()
          self.conv_RU5_1_depth = nn.Conv2d(64, 64, 3, padding=1, bias=False)
          self.bn_RU5_1_depth = nn.BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True)
          self.relu_RU5_1_depth = nn.PReLU()
          self.conv_RU5_2_depth = nn.Conv2d(64, 64, 3, padding=1, bias=False)
          self.bn_RU5_2_depth = nn.BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True)

          self.relu_RU_end5_depth = nn.PReLU()
          # self.tran1 = nn.AdaptiveAvgPool2d((1, 1))

          self.gate_rear4 = nn.Conv2d(256, 128, 1, padding=0, bias=False)
          self.bn4 = nn.BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True)
          self.relu4_3 = nn.ReLU(inplace=True)
          self.sconv4 = nn.Conv2d(128, 128, 1, padding=0, bias=False)
          self.sconv_bn4 = nn.BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True)
          self.sconv_relu4 = nn.ReLU(inplace=True)
          self.cat_feature_out4 = nn.Sequential(nn.Conv2d(128, 32, 3, padding=1), nn.ReLU(inplace=True),
                                              nn.Conv2d(32, 1, 1, padding=0)
                                             )
          
          self.relu_RU4_img = nn.PReLU()
          self.conv_RU4_1_img = nn.Conv2d(128, 128, 3, padding=1, bias=False)
          self.bn_RU4_1_img = nn.BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True)
          self.relu_RU4_1_img = nn.PReLU()
          self.conv_RU4_2_img = nn.Conv2d(128, 128, 3, padding=1, bias=False)
          self.bn_RU4_2_img = nn.BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True)

          self.relu_RU_end4_img = nn.PReLU()

          self.relu_RU4_depth = nn.PReLU()
          self.conv_RU4_1_depth = nn.Conv2d(128, 128, 3, padding=1, bias=False)
          self.bn_RU4_1_depth = nn.BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True)
          self.relu_RU4_1_depth = nn.PReLU()
          self.conv_RU4_2_depth = nn.Conv2d(128, 128, 3, padding=1, bias=False)
          self.bn_RU4_2_depth = nn.BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True)

          self.relu_RU_end4_depth = nn.PReLU()

          self.gate_rear3 = nn.Conv2d(512, 256, 1, padding=0, bias=False)
          self.bn3 = nn.BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True)
          self.relu3_3 = nn.ReLU(inplace=True)
          self.sconv3 = nn.Conv2d(256, 256, 1, padding=0, bias=False)
          self.sconv_bn3 = nn.BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True)
          self.sconv_relu3 = nn.ReLU(inplace=True)
          self.cat_feature_out3 = nn.Sequential(nn.Conv2d(256, 64, 3, padding=1), nn.ReLU(inplace=True),
                                              nn.Conv2d(64, 1, 1, padding=0)
                                             )

          self.relu_RU3_img = nn.PReLU()
          self.conv_RU3_1_img = nn.Conv2d(256, 256, 3, padding=1, bias=False)
          self.bn_RU3_1_img = nn.BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True)
          self.relu_RU3_1_img = nn.PReLU()
          self.conv_RU3_2_img = nn.Conv2d(256, 256, 3, padding=1, bias=False)
          self.bn_RU3_2_img = nn.BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True)

          self.relu_RU_end3_img = nn.PReLU()

          self.relu_RU3_depth = nn.PReLU()
          self.conv_RU3_1_depth = nn.Conv2d(256, 256, 3, padding=1, bias=False)
          self.bn_RU3_1_depth = nn.BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True)
          self.relu_RU3_1_depth = nn.PReLU()
          self.conv_RU3_2_depth = nn.Conv2d(256, 256, 3, padding=1, bias=False)
          self.bn_RU3_2_depth = nn.BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True)

          self.relu_RU_end3_depth = nn.PReLU()

          self.img_con2= nn.Sequential(nn.Conv2d(512, 512, 3, padding=1), nn.ReLU(inplace=True),
                                     nn.Conv2d(512, 512, 1, padding=0), nn.ReLU(inplace=True),)
          self.depth_con2 = nn.Sequential(nn.Conv2d(512, 512, 3, padding=1), nn.ReLU(inplace=True),
                                     nn.Conv2d(512, 512, 1, padding=0), nn.ReLU(inplace=True), )
          self.gate_rear2 = nn.Conv2d(1024, 512, 1, padding=0, bias=False)
          self.bn2 = nn.BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True)
          self.relu4_4 = nn.ReLU(inplace=True)
          self.sconv4_4 = nn.Conv2d(512, 512, 1, padding=0, bias=False)
          self.sconv_bn4_4 = nn.BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True)
          self.sconv_relu4_4 = nn.ReLU(inplace=True)

          self.img_con1= nn.Sequential(nn.Conv2d(512, 512, 3, padding=1), nn.ReLU(inplace=True),
                                     nn.Conv2d(512, 512, 1, padding=0), nn.ReLU(inplace=True),)
          self.depth_con1 = nn.Sequential(nn.Conv2d(512, 512, 3, padding=1), nn.ReLU(inplace=True),
                                     nn.Conv2d(512, 512, 1, padding=0), nn.ReLU(inplace=True), )
          self.gate_rear5_4 = nn.Conv2d(1024, 512, 1, padding=0, bias=False)
          self.bn5_4 = nn.BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True)
          self.relu5_4 = nn.ReLU(inplace=True)
          self.sconv5_4 = nn.Conv2d(512, 512, 1, padding=0, bias=False)
          self.sconv_bn5_4 = nn.BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True)
          self.sconv_relu5_4 = nn.ReLU(inplace=True)
     
     def forward(self, h1, h2, h3, h4, h5, d1, d2, d3, d4, d5):
          ##### GSFM for Conv1_2 #####
          #concat h1 and d1
          cat_feature1 = torch.cat((h1, d1), 1)  #torch.Size([10, 128, 256, 256])
          # print (cat_feature1.shape)
          #Adaptation layer
          cat_feature1 = self.bn1(self.gate_rear1(cat_feature1))  #torch.Size([10, 64, 256, 256])
          #Integration layer
          cat_feature1 = self.relu1_2(cat_feature1)
          cat_feature1_in = self.sconv_relu1(self.sconv_bn1(self.sconv1_2(cat_feature1)))
          gate_out1 = torch.sigmoid(self.cat_feature_out1(cat_feature1_in))
          #RU RGB
          RGB_feature = h1
          RGB_feature = self.relu_RU5_img(RGB_feature)
          RGB_feature = self.relu_RU5_1_img(self.bn_RU5_1_img(self.conv_RU5_1_img(RGB_feature)))
          RGB_feature = self.bn_RU5_2_img(self.conv_RU5_2_img(RGB_feature))
          RGB_feature1 = h1 + RGB_feature
          RGB_feature1 = self.relu_RU_end5_img(RGB_feature1)
          #RU Depth
          Depth_feature = d1
          Depth_feature = self.relu_RU5_depth(Depth_feature)
          Depth_feature = self.relu_RU5_1_depth(self.bn_RU5_1_depth(self.conv_RU5_1_depth(Depth_feature)))
          Depth_feature = self.bn_RU5_2_depth(self.conv_RU5_2_depth(Depth_feature))
          Depth_feature1 = d1 + Depth_feature
          Depth_feature1 = self.relu_RU_end5_depth(Depth_feature1)
          #Gate
          RGB_feature1 = torch.mul(gate_out1,RGB_feature1)
          Depth_feature1 = torch.mul(gate_out1, Depth_feature1)
          feature1 = torch.cat((RGB_feature1,Depth_feature1),1) #torch.Size([10, 128, 256, 256])
          # feature1 = self.tran1(feature1)
          # feature1 = F.interpolate(feature1, size=[128, 128], mode="bilinear")
          feature1 = F.interpolate(feature1, size=[128, 128], mode="bilinear", align_corners=True)  #torch.Size([10, 128, 128, 128])

          ##### GSFM for Conv2_2 #####
          #concat h2 and d2
          cat_feature2 = torch.cat((h2, d2), 1)  #torch.Size([10, 256, 128, 128])
          #Adaptation layer
          cat_feature2 = self.bn4(self.gate_rear4(cat_feature2))  #torch.Size([10, 128, 128, 128])
          cat_feature2 = cat_feature2 + feature1
          #Integration layer
          cat_feature2 = self.relu4_3(cat_feature2)
          cat_feature2_in = self.sconv_relu4(self.sconv_bn4(self.sconv4(cat_feature2)))
          gate_out2 = torch.sigmoid(self.cat_feature_out4(cat_feature2_in))
          # RU RGB
          RGB_feature = h2
          RGB_feature = self.relu_RU4_img(RGB_feature)
          RGB_feature = self.relu_RU4_1_img(self.bn_RU4_1_img(self.conv_RU4_1_img(RGB_feature)))
          RGB_feature = self.bn_RU4_2_img(self.conv_RU4_2_img(RGB_feature))
          RGB_feature2 = h2 + RGB_feature
          RGB_feature2 = self.relu_RU_end4_img(RGB_feature2)
          # RU Depth
          Depth_feature = d2
          Depth_feature = self.relu_RU4_depth(Depth_feature)
          Depth_feature = self.relu_RU4_1_depth(self.bn_RU4_1_depth(self.conv_RU4_1_depth(Depth_feature)))
          Depth_feature = self.bn_RU4_2_depth(self.conv_RU4_2_depth(Depth_feature))
          Depth_feature2 = d2 + Depth_feature
          Depth_feature2 = self.relu_RU_end4_depth(Depth_feature2)
          # out
          RGB_feature2 = torch.mul(gate_out2,RGB_feature2)
          Depth_feature2 = torch.mul(gate_out2, Depth_feature2)
          feature2 = torch.cat((RGB_feature2,Depth_feature2),1)
          feature2 = F.interpolate(feature2, size=[64, 64], mode="bilinear", align_corners=True)  #torch.Size([10, 256, 64, 64])

          ##### GSFM for Conv3_3 #####
          #concat h3 and d3
          cat_feature3 = torch.cat((h3, d3), 1)  #torch.Size([10, 512, 64, 64])
          #Adaptation layer
          cat_feature3 = self.bn3(self.gate_rear3(cat_feature3))
          cat_feature3 = cat_feature3 + feature2
          #Integration layer
          cat_feature3 = self.relu3_3(cat_feature3)
          cat_feature3_in = self.sconv_relu3(self.sconv_bn3(self.sconv3(cat_feature3)))
          gate_out3 = torch.sigmoid(self.cat_feature_out3(cat_feature3_in))
          # RU RGB
          RGB_feature = h3
          RGB_feature = self.relu_RU3_img(RGB_feature)
          RGB_feature = self.relu_RU3_1_img(self.bn_RU3_1_img(self.conv_RU3_1_img(RGB_feature)))
          RGB_feature = self.bn_RU3_2_img(self.conv_RU3_2_img(RGB_feature))
          RGB_feature3 = h3 + RGB_feature
          RGB_feature3 = self.relu_RU_end3_img(RGB_feature3)
          # RU Depth
          Depth_feature = d3
          Depth_feature = self.relu_RU3_depth(Depth_feature)
          Depth_feature = self.relu_RU3_1_depth(self.bn_RU3_1_depth(self.conv_RU3_1_depth(Depth_feature)))
          Depth_feature = self.bn_RU3_2_depth(self.conv_RU3_2_depth(Depth_feature))
          Depth_feature3 = d3 + Depth_feature
          Depth_feature3 = self.relu_RU_end3_depth(Depth_feature3)
          # out
          RGB_feature3 = torch.mul(gate_out3,RGB_feature3)
          Depth_feature3 = torch.mul(gate_out3, Depth_feature3)
          feature3 = torch.cat((RGB_feature3,Depth_feature3),1)  #torch.Size([10, 512, 64, 64])
          feature3 = F.interpolate(feature3, size=[32, 32], mode="bilinear", align_corners=True)  #torch.Size([10, 512, 32, 32])

          ##### GSFM for Conv4_4 #####
          #concat h4 and d4
          img_feature4 = self.img_con2(h4)
          depth_feature4 = self.depth_con2(d4)
          cat_feature4 = torch.cat((img_feature4,depth_feature4),1)
          #Adaptation layer
          cat_feature4 = self.bn2(self.gate_rear2(cat_feature4))  #torch.Size([10, 512, 32, 32])
          feature4 = cat_feature4 + feature3
          #Integration layer
          feature4 = self.relu4_4(feature4)
          feature4 = self.sconv_relu4_4(self.sconv_bn4_4(self.sconv4_4(feature4)))  #torch.Size([10, 512, 32, 32])
          feature4 = F.interpolate(feature4, size=[16, 16], mode="bilinear", align_corners=True)  #torch.Size([10, 512, 16, 16])
          
          ##### GSFM for Conv5_4 #####
          #concat h5 and d5
          img_feature5 = self.img_con1(h5)
          depth_feature5 = self.depth_con1(d5)
          cat_feature5 = torch.cat((img_feature5,depth_feature5),1)
          #Adaptation layer
          cat_feature5 = self.bn5_4(self.gate_rear5_4(cat_feature5))  #torch.Size([10, 512, 16, 16])
          feature5 = cat_feature5 + feature4
          #Integration layer
          feature5 = self.relu5_4(feature5)
          out_rgbdepth = self.sconv_relu5_4(self.sconv_bn5_4(self.sconv5_4(feature5)))  #torch.Size([10, 512, 16, 16])
          return out_rgbdepth


class Fusion_rgbmpt(nn.Module):
     def __init__(self, bias=True):
          super(Fusion_rgbmpt, self).__init__()
          self.gate_rear1 = nn.Conv2d(128, 64, 1, padding=0, bias=False)
          self.bn1 = nn.BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True)

          self.relu1_2 = nn.ReLU(inplace=True)
          self.sconv1_2 = nn.Conv2d(64, 64, 1, padding=0, bias=False)
          self.sconv_bn1 = nn.BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True)
          self.sconv_relu1 = nn.ReLU(inplace=True)
          self.cat_feature_out1 = nn.Sequential(nn.Conv2d(64, 16, 3, padding=1), nn.ReLU(inplace=True),
                                              nn.Conv2d(16, 1, 1, padding=0)
                                             )
          self.relu_RU5_img = nn.PReLU()
          self.conv_RU5_1_img = nn.Conv2d(64, 64, 3, padding=1, bias=False)
          self.bn_RU5_1_img = nn.BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True)
          self.relu_RU5_1_img = nn.PReLU()
          self.conv_RU5_2_img = nn.Conv2d(64, 64, 3, padding=1, bias=False)
          self.bn_RU5_2_img = nn.BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True)

          self.relu_RU_end5_img = nn.PReLU()

          self.relu_RU5_depth = nn.PReLU()
          self.conv_RU5_1_depth = nn.Conv2d(64, 64, 3, padding=1, bias=False)
          self.bn_RU5_1_depth = nn.BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True)
          self.relu_RU5_1_depth = nn.PReLU()
          self.conv_RU5_2_depth = nn.Conv2d(64, 64, 3, padding=1, bias=False)
          self.bn_RU5_2_depth = nn.BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True)

          self.relu_RU_end5_depth = nn.PReLU()
          # self.tran1 = nn.AdaptiveAvgPool2d((1, 1))

          self.gate_rear4 = nn.Conv2d(256, 128, 1, padding=0, bias=False)
          self.bn4 = nn.BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True)
          self.relu4_3 = nn.ReLU(inplace=True)
          self.sconv4 = nn.Conv2d(128, 128, 1, padding=0, bias=False)
          self.sconv_bn4 = nn.BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True)
          self.sconv_relu4 = nn.ReLU(inplace=True)
          self.cat_feature_out4 = nn.Sequential(nn.Conv2d(128, 32, 3, padding=1), nn.ReLU(inplace=True),
                                              nn.Conv2d(32, 1, 1, padding=0)
                                             )
          
          self.relu_RU4_img = nn.PReLU()
          self.conv_RU4_1_img = nn.Conv2d(128, 128, 3, padding=1, bias=False)
          self.bn_RU4_1_img = nn.BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True)
          self.relu_RU4_1_img = nn.PReLU()
          self.conv_RU4_2_img = nn.Conv2d(128, 128, 3, padding=1, bias=False)
          self.bn_RU4_2_img = nn.BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True)

          self.relu_RU_end4_img = nn.PReLU()

          self.relu_RU4_depth = nn.PReLU()
          self.conv_RU4_1_depth = nn.Conv2d(128, 128, 3, padding=1, bias=False)
          self.bn_RU4_1_depth = nn.BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True)
          self.relu_RU4_1_depth = nn.PReLU()
          self.conv_RU4_2_depth = nn.Conv2d(128, 128, 3, padding=1, bias=False)
          self.bn_RU4_2_depth = nn.BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True)

          self.relu_RU_end4_depth = nn.PReLU()

          self.gate_rear3 = nn.Conv2d(512, 256, 1, padding=0, bias=False)
          self.bn3 = nn.BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True)
          self.relu3_3 = nn.ReLU(inplace=True)
          self.sconv3 = nn.Conv2d(256, 256, 1, padding=0, bias=False)
          self.sconv_bn3 = nn.BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True)
          self.sconv_relu3 = nn.ReLU(inplace=True)
          self.cat_feature_out3 = nn.Sequential(nn.Conv2d(256, 64, 3, padding=1), nn.ReLU(inplace=True),
                                              nn.Conv2d(64, 1, 1, padding=0)
                                             )

          self.relu_RU3_img = nn.PReLU()
          self.conv_RU3_1_img = nn.Conv2d(256, 256, 3, padding=1, bias=False)
          self.bn_RU3_1_img = nn.BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True)
          self.relu_RU3_1_img = nn.PReLU()
          self.conv_RU3_2_img = nn.Conv2d(256, 256, 3, padding=1, bias=False)
          self.bn_RU3_2_img = nn.BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True)

          self.relu_RU_end3_img = nn.PReLU()

          self.relu_RU3_depth = nn.PReLU()
          self.conv_RU3_1_depth = nn.Conv2d(256, 256, 3, padding=1, bias=False)
          self.bn_RU3_1_depth = nn.BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True)
          self.relu_RU3_1_depth = nn.PReLU()
          self.conv_RU3_2_depth = nn.Conv2d(256, 256, 3, padding=1, bias=False)
          self.bn_RU3_2_depth = nn.BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True)

          self.relu_RU_end3_depth = nn.PReLU()

          self.img_con2= nn.Sequential(nn.Conv2d(512, 512, 3, padding=1), nn.ReLU(inplace=True),
                                     nn.Conv2d(512, 512, 1, padding=0), nn.ReLU(inplace=True),)
          self.depth_con2 = nn.Sequential(nn.Conv2d(512, 512, 3, padding=1), nn.ReLU(inplace=True),
                                     nn.Conv2d(512, 512, 1, padding=0), nn.ReLU(inplace=True), )
          self.gate_rear2 = nn.Conv2d(1024, 512, 1, padding=0, bias=False)
          self.bn2 = nn.BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True)
          self.relu4_4 = nn.ReLU(inplace=True)
          self.sconv4_4 = nn.Conv2d(512, 512, 1, padding=0, bias=False)
          self.sconv_bn4_4 = nn.BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True)
          self.sconv_relu4_4 = nn.ReLU(inplace=True)

          self.img_con1= nn.Sequential(nn.Conv2d(512, 512, 3, padding=1), nn.ReLU(inplace=True),
                                     nn.Conv2d(512, 512, 1, padding=0), nn.ReLU(inplace=True),)
          self.depth_con1 = nn.Sequential(nn.Conv2d(512, 512, 3, padding=1), nn.ReLU(inplace=True),
                                     nn.Conv2d(512, 512, 1, padding=0), nn.ReLU(inplace=True), )
          self.gate_rear5_4 = nn.Conv2d(1024, 512, 1, padding=0, bias=False)
          self.bn5_4 = nn.BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True)
          self.relu5_4 = nn.ReLU(inplace=True)
          self.sconv5_4 = nn.Conv2d(512, 512, 1, padding=0, bias=False)
          self.sconv_bn5_4 = nn.BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True)
          self.sconv_relu5_4 = nn.ReLU(inplace=True)
     
     def forward(self, h1, h2, h3, h4, h5, d1, d2, d3, d4, d5):
          ##### GSFM for Conv1_2 #####
          #concat h1 and d1
          cat_feature1 = torch.cat((h1, d1), 1)  #torch.Size([10, 128, 256, 256])
          # print (cat_feature1.shape)
          #Adaptation layer
          cat_feature1 = self.bn1(self.gate_rear1(cat_feature1))  #torch.Size([10, 64, 256, 256])
          #Integration layer
          cat_feature1 = self.relu1_2(cat_feature1)
          cat_feature1_in = self.sconv_relu1(self.sconv_bn1(self.sconv1_2(cat_feature1)))
          gate_out1 = torch.sigmoid(self.cat_feature_out1(cat_feature1_in))
          #RU RGB
          RGB_feature = h1
          RGB_feature = self.relu_RU5_img(RGB_feature)
          RGB_feature = self.relu_RU5_1_img(self.bn_RU5_1_img(self.conv_RU5_1_img(RGB_feature)))
          RGB_feature = self.bn_RU5_2_img(self.conv_RU5_2_img(RGB_feature))
          RGB_feature1 = h1 + RGB_feature
          RGB_feature1 = self.relu_RU_end5_img(RGB_feature1)
          #RU Depth
          Depth_feature = d1
          Depth_feature = self.relu_RU5_depth(Depth_feature)
          Depth_feature = self.relu_RU5_1_depth(self.bn_RU5_1_depth(self.conv_RU5_1_depth(Depth_feature)))
          Depth_feature = self.bn_RU5_2_depth(self.conv_RU5_2_depth(Depth_feature))
          Depth_feature1 = d1 + Depth_feature
          Depth_feature1 = self.relu_RU_end5_depth(Depth_feature1)
          #Gate
          RGB_feature1 = torch.mul(gate_out1,RGB_feature1)
          Depth_feature1 = torch.mul(gate_out1, Depth_feature1)
          feature1 = torch.cat((RGB_feature1,Depth_feature1),1) #torch.Size([10, 128, 256, 256])
          # feature1 = self.tran1(feature1)
          # feature1 = F.interpolate(feature1, size=[128, 128], mode="bilinear")
          feature1 = F.interpolate(feature1, size=[128, 128], mode="bilinear", align_corners=True)  #torch.Size([10, 128, 128, 128])

          ##### GSFM for Conv2_2 #####
          #concat h2 and d2
          cat_feature2 = torch.cat((h2, d2), 1)  #torch.Size([10, 256, 128, 128])
          #Adaptation layer
          cat_feature2 = self.bn4(self.gate_rear4(cat_feature2))  #torch.Size([10, 128, 128, 128])
          cat_feature2 = cat_feature2 + feature1
          #Integration layer
          cat_feature2 = self.relu4_3(cat_feature2)
          cat_feature2_in = self.sconv_relu4(self.sconv_bn4(self.sconv4(cat_feature2)))
          gate_out2 = torch.sigmoid(self.cat_feature_out4(cat_feature2_in))
          # RU RGB
          RGB_feature = h2
          RGB_feature = self.relu_RU4_img(RGB_feature)
          RGB_feature = self.relu_RU4_1_img(self.bn_RU4_1_img(self.conv_RU4_1_img(RGB_feature)))
          RGB_feature = self.bn_RU4_2_img(self.conv_RU4_2_img(RGB_feature))
          RGB_feature2 = h2 + RGB_feature
          RGB_feature2 = self.relu_RU_end4_img(RGB_feature2)
          # RU Depth
          Depth_feature = d2
          Depth_feature = self.relu_RU4_depth(Depth_feature)
          Depth_feature = self.relu_RU4_1_depth(self.bn_RU4_1_depth(self.conv_RU4_1_depth(Depth_feature)))
          Depth_feature = self.bn_RU4_2_depth(self.conv_RU4_2_depth(Depth_feature))
          Depth_feature2 = d2 + Depth_feature
          Depth_feature2 = self.relu_RU_end4_depth(Depth_feature2)
          # out
          RGB_feature2 = torch.mul(gate_out2,RGB_feature2)
          Depth_feature2 = torch.mul(gate_out2, Depth_feature2)
          feature2 = torch.cat((RGB_feature2,Depth_feature2),1)
          feature2 = F.interpolate(feature2, size=[64, 64], mode="bilinear", align_corners=True)  #torch.Size([10, 256, 64, 64])

          ##### GSFM for Conv3_3 #####
          #concat h3 and d3
          cat_feature3 = torch.cat((h3, d3), 1)  #torch.Size([10, 512, 64, 64])
          #Adaptation layer
          cat_feature3 = self.bn3(self.gate_rear3(cat_feature3))
          cat_feature3 = cat_feature3 + feature2
          #Integration layer
          cat_feature3 = self.relu3_3(cat_feature3)
          cat_feature3_in = self.sconv_relu3(self.sconv_bn3(self.sconv3(cat_feature3)))
          gate_out3 = torch.sigmoid(self.cat_feature_out3(cat_feature3_in))
          # RU RGB
          RGB_feature = h3
          RGB_feature = self.relu_RU3_img(RGB_feature)
          RGB_feature = self.relu_RU3_1_img(self.bn_RU3_1_img(self.conv_RU3_1_img(RGB_feature)))
          RGB_feature = self.bn_RU3_2_img(self.conv_RU3_2_img(RGB_feature))
          RGB_feature3 = h3 + RGB_feature
          RGB_feature3 = self.relu_RU_end3_img(RGB_feature3)
          # RU Depth
          Depth_feature = d3
          Depth_feature = self.relu_RU3_depth(Depth_feature)
          Depth_feature = self.relu_RU3_1_depth(self.bn_RU3_1_depth(self.conv_RU3_1_depth(Depth_feature)))
          Depth_feature = self.bn_RU3_2_depth(self.conv_RU3_2_depth(Depth_feature))
          Depth_feature3 = d3 + Depth_feature
          Depth_feature3 = self.relu_RU_end3_depth(Depth_feature3)
          # out
          RGB_feature3 = torch.mul(gate_out3,RGB_feature3)
          Depth_feature3 = torch.mul(gate_out3, Depth_feature3)
          feature3 = torch.cat((RGB_feature3,Depth_feature3),1)  #torch.Size([10, 512, 64, 64])
          feature3 = F.interpolate(feature3, size=[32, 32], mode="bilinear", align_corners=True)  #torch.Size([10, 512, 32, 32])

          ##### GSFM for Conv4_4 #####
          #concat h4 and d4
          img_feature4 = self.img_con2(h4)
          depth_feature4 = self.depth_con2(d4)
          cat_feature4 = torch.cat((img_feature4,depth_feature4),1)
          #Adaptation layer
          cat_feature4 = self.bn2(self.gate_rear2(cat_feature4))  #torch.Size([10, 512, 32, 32])
          feature4 = cat_feature4 + feature3
          #Integration layer
          feature4 = self.relu4_4(feature4)
          feature4 = self.sconv_relu4_4(self.sconv_bn4_4(self.sconv4_4(feature4)))  #torch.Size([10, 512, 32, 32])
          feature4 = F.interpolate(feature4, size=[16, 16], mode="bilinear", align_corners=True)  #torch.Size([10, 512, 16, 16])
          
          ##### GSFM for Conv5_4 #####
          #concat h5 and d5
          img_feature5 = self.img_con1(h5)
          depth_feature5 = self.depth_con1(d5)
          cat_feature5 = torch.cat((img_feature5,depth_feature5),1)
          #Adaptation layer
          cat_feature5 = self.bn5_4(self.gate_rear5_4(cat_feature5))  #torch.Size([10, 512, 16, 16])
          feature5 = cat_feature5 + feature4
          #Integration layer
          feature5 = self.relu5_4(feature5)
          out_rgbmpt = self.sconv_relu5_4(self.sconv_bn5_4(self.sconv5_4(feature5)))  #torch.Size([10, 512, 16, 16])
          return out_rgbmpt




class Fusion_depthmpt(nn.Module):
     def __init__(self, bias=True):
          super(Fusion_depthmpt, self).__init__()
          self.gate_rear1 = nn.Conv2d(128, 64, 1, padding=0, bias=False)
          self.bn1 = nn.BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True)

          self.relu1_2 = nn.ReLU(inplace=True)
          self.sconv1_2 = nn.Conv2d(64, 64, 1, padding=0, bias=False)
          self.sconv_bn1 = nn.BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True)
          self.sconv_relu1 = nn.ReLU(inplace=True)
          self.cat_feature_out1 = nn.Sequential(nn.Conv2d(64, 16, 3, padding=1), nn.ReLU(inplace=True),
                                              nn.Conv2d(16, 1, 1, padding=0)
                                             )
          self.relu_RU5_img = nn.PReLU()
          self.conv_RU5_1_img = nn.Conv2d(64, 64, 3, padding=1, bias=False)
          self.bn_RU5_1_img = nn.BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True)
          self.relu_RU5_1_img = nn.PReLU()
          self.conv_RU5_2_img = nn.Conv2d(64, 64, 3, padding=1, bias=False)
          self.bn_RU5_2_img = nn.BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True)

          self.relu_RU_end5_img = nn.PReLU()

          self.relu_RU5_depth = nn.PReLU()
          self.conv_RU5_1_depth = nn.Conv2d(64, 64, 3, padding=1, bias=False)
          self.bn_RU5_1_depth = nn.BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True)
          self.relu_RU5_1_depth = nn.PReLU()
          self.conv_RU5_2_depth = nn.Conv2d(64, 64, 3, padding=1, bias=False)
          self.bn_RU5_2_depth = nn.BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True)

          self.relu_RU_end5_depth = nn.PReLU()
          # self.tran1 = nn.AdaptiveAvgPool2d((1, 1))

          self.gate_rear4 = nn.Conv2d(256, 128, 1, padding=0, bias=False)
          self.bn4 = nn.BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True)
          self.relu4_3 = nn.ReLU(inplace=True)
          self.sconv4 = nn.Conv2d(128, 128, 1, padding=0, bias=False)
          self.sconv_bn4 = nn.BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True)
          self.sconv_relu4 = nn.ReLU(inplace=True)
          self.cat_feature_out4 = nn.Sequential(nn.Conv2d(128, 32, 3, padding=1), nn.ReLU(inplace=True),
                                              nn.Conv2d(32, 1, 1, padding=0)
                                             )
          
          self.relu_RU4_img = nn.PReLU()
          self.conv_RU4_1_img = nn.Conv2d(128, 128, 3, padding=1, bias=False)
          self.bn_RU4_1_img = nn.BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True)
          self.relu_RU4_1_img = nn.PReLU()
          self.conv_RU4_2_img = nn.Conv2d(128, 128, 3, padding=1, bias=False)
          self.bn_RU4_2_img = nn.BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True)

          self.relu_RU_end4_img = nn.PReLU()

          self.relu_RU4_depth = nn.PReLU()
          self.conv_RU4_1_depth = nn.Conv2d(128, 128, 3, padding=1, bias=False)
          self.bn_RU4_1_depth = nn.BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True)
          self.relu_RU4_1_depth = nn.PReLU()
          self.conv_RU4_2_depth = nn.Conv2d(128, 128, 3, padding=1, bias=False)
          self.bn_RU4_2_depth = nn.BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True)

          self.relu_RU_end4_depth = nn.PReLU()

          self.gate_rear3 = nn.Conv2d(512, 256, 1, padding=0, bias=False)
          self.bn3 = nn.BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True)
          self.relu3_3 = nn.ReLU(inplace=True)
          self.sconv3 = nn.Conv2d(256, 256, 1, padding=0, bias=False)
          self.sconv_bn3 = nn.BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True)
          self.sconv_relu3 = nn.ReLU(inplace=True)
          self.cat_feature_out3 = nn.Sequential(nn.Conv2d(256, 64, 3, padding=1), nn.ReLU(inplace=True),
                                              nn.Conv2d(64, 1, 1, padding=0)
                                             )

          self.relu_RU3_img = nn.PReLU()
          self.conv_RU3_1_img = nn.Conv2d(256, 256, 3, padding=1, bias=False)
          self.bn_RU3_1_img = nn.BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True)
          self.relu_RU3_1_img = nn.PReLU()
          self.conv_RU3_2_img = nn.Conv2d(256, 256, 3, padding=1, bias=False)
          self.bn_RU3_2_img = nn.BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True)

          self.relu_RU_end3_img = nn.PReLU()

          self.relu_RU3_depth = nn.PReLU()
          self.conv_RU3_1_depth = nn.Conv2d(256, 256, 3, padding=1, bias=False)
          self.bn_RU3_1_depth = nn.BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True)
          self.relu_RU3_1_depth = nn.PReLU()
          self.conv_RU3_2_depth = nn.Conv2d(256, 256, 3, padding=1, bias=False)
          self.bn_RU3_2_depth = nn.BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True)

          self.relu_RU_end3_depth = nn.PReLU()

          self.img_con2= nn.Sequential(nn.Conv2d(512, 512, 3, padding=1), nn.ReLU(inplace=True),
                                     nn.Conv2d(512, 512, 1, padding=0), nn.ReLU(inplace=True),)
          self.depth_con2 = nn.Sequential(nn.Conv2d(512, 512, 3, padding=1), nn.ReLU(inplace=True),
                                     nn.Conv2d(512, 512, 1, padding=0), nn.ReLU(inplace=True), )
          self.gate_rear2 = nn.Conv2d(1024, 512, 1, padding=0, bias=False)
          self.bn2 = nn.BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True)
          self.relu4_4 = nn.ReLU(inplace=True)
          self.sconv4_4 = nn.Conv2d(512, 512, 1, padding=0, bias=False)
          self.sconv_bn4_4 = nn.BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True)
          self.sconv_relu4_4 = nn.ReLU(inplace=True)

          self.img_con1= nn.Sequential(nn.Conv2d(512, 512, 3, padding=1), nn.ReLU(inplace=True),
                                     nn.Conv2d(512, 512, 1, padding=0), nn.ReLU(inplace=True),)
          self.depth_con1 = nn.Sequential(nn.Conv2d(512, 512, 3, padding=1), nn.ReLU(inplace=True),
                                     nn.Conv2d(512, 512, 1, padding=0), nn.ReLU(inplace=True), )
          self.gate_rear5_4 = nn.Conv2d(1024, 512, 1, padding=0, bias=False)
          self.bn5_4 = nn.BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True)
          self.relu5_4 = nn.ReLU(inplace=True)
          self.sconv5_4 = nn.Conv2d(512, 512, 1, padding=0, bias=False)
          self.sconv_bn5_4 = nn.BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True)
          self.sconv_relu5_4 = nn.ReLU(inplace=True)
     
     def forward(self, h1, h2, h3, h4, h5, d1, d2, d3, d4, d5):
          ##### GSFM for Conv1_2 #####
          #concat h1 and d1
          cat_feature1 = torch.cat((h1, d1), 1)  #torch.Size([10, 128, 256, 256])
          # print (cat_feature1.shape)
          #Adaptation layer
          cat_feature1 = self.bn1(self.gate_rear1(cat_feature1))  #torch.Size([10, 64, 256, 256])
          #Integration layer
          cat_feature1 = self.relu1_2(cat_feature1)
          cat_feature1_in = self.sconv_relu1(self.sconv_bn1(self.sconv1_2(cat_feature1)))
          gate_out1 = torch.sigmoid(self.cat_feature_out1(cat_feature1_in))
          #RU RGB
          RGB_feature = h1
          RGB_feature = self.relu_RU5_img(RGB_feature)
          RGB_feature = self.relu_RU5_1_img(self.bn_RU5_1_img(self.conv_RU5_1_img(RGB_feature)))
          RGB_feature = self.bn_RU5_2_img(self.conv_RU5_2_img(RGB_feature))
          RGB_feature1 = h1 + RGB_feature
          RGB_feature1 = self.relu_RU_end5_img(RGB_feature1)
          #RU Depth
          Depth_feature = d1
          Depth_feature = self.relu_RU5_depth(Depth_feature)
          Depth_feature = self.relu_RU5_1_depth(self.bn_RU5_1_depth(self.conv_RU5_1_depth(Depth_feature)))
          Depth_feature = self.bn_RU5_2_depth(self.conv_RU5_2_depth(Depth_feature))
          Depth_feature1 = d1 + Depth_feature
          Depth_feature1 = self.relu_RU_end5_depth(Depth_feature1)
          #Gate
          RGB_feature1 = torch.mul(gate_out1,RGB_feature1)
          Depth_feature1 = torch.mul(gate_out1, Depth_feature1)
          feature1 = torch.cat((RGB_feature1,Depth_feature1),1) #torch.Size([10, 128, 256, 256])
          # feature1 = self.tran1(feature1)
          # feature1 = F.interpolate(feature1, size=[128, 128], mode="bilinear")
          feature1 = F.interpolate(feature1, size=[128, 128], mode="bilinear", align_corners=True)  #torch.Size([10, 128, 128, 128])

          ##### GSFM for Conv2_2 #####
          #concat h2 and d2
          cat_feature2 = torch.cat((h2, d2), 1)  #torch.Size([10, 256, 128, 128])
          #Adaptation layer
          cat_feature2 = self.bn4(self.gate_rear4(cat_feature2))  #torch.Size([10, 128, 128, 128])
          cat_feature2 = cat_feature2 + feature1
          #Integration layer
          cat_feature2 = self.relu4_3(cat_feature2)
          cat_feature2_in = self.sconv_relu4(self.sconv_bn4(self.sconv4(cat_feature2)))
          gate_out2 = torch.sigmoid(self.cat_feature_out4(cat_feature2_in))
          # RU RGB
          RGB_feature = h2
          RGB_feature = self.relu_RU4_img(RGB_feature)
          RGB_feature = self.relu_RU4_1_img(self.bn_RU4_1_img(self.conv_RU4_1_img(RGB_feature)))
          RGB_feature = self.bn_RU4_2_img(self.conv_RU4_2_img(RGB_feature))
          RGB_feature2 = h2 + RGB_feature
          RGB_feature2 = self.relu_RU_end4_img(RGB_feature2)
          # RU Depth
          Depth_feature = d2
          Depth_feature = self.relu_RU4_depth(Depth_feature)
          Depth_feature = self.relu_RU4_1_depth(self.bn_RU4_1_depth(self.conv_RU4_1_depth(Depth_feature)))
          Depth_feature = self.bn_RU4_2_depth(self.conv_RU4_2_depth(Depth_feature))
          Depth_feature2 = d2 + Depth_feature
          Depth_feature2 = self.relu_RU_end4_depth(Depth_feature2)
          # out
          RGB_feature2 = torch.mul(gate_out2,RGB_feature2)
          Depth_feature2 = torch.mul(gate_out2, Depth_feature2)
          feature2 = torch.cat((RGB_feature2,Depth_feature2),1)
          feature2 = F.interpolate(feature2, size=[64, 64], mode="bilinear", align_corners=True)  #torch.Size([10, 256, 64, 64])

          ##### GSFM for Conv3_3 #####
          #concat h3 and d3
          cat_feature3 = torch.cat((h3, d3), 1)  #torch.Size([10, 512, 64, 64])
          #Adaptation layer
          cat_feature3 = self.bn3(self.gate_rear3(cat_feature3))
          cat_feature3 = cat_feature3 + feature2
          #Integration layer
          cat_feature3 = self.relu3_3(cat_feature3)
          cat_feature3_in = self.sconv_relu3(self.sconv_bn3(self.sconv3(cat_feature3)))
          gate_out3 = torch.sigmoid(self.cat_feature_out3(cat_feature3_in))
          # RU RGB
          RGB_feature = h3
          RGB_feature = self.relu_RU3_img(RGB_feature)
          RGB_feature = self.relu_RU3_1_img(self.bn_RU3_1_img(self.conv_RU3_1_img(RGB_feature)))
          RGB_feature = self.bn_RU3_2_img(self.conv_RU3_2_img(RGB_feature))
          RGB_feature3 = h3 + RGB_feature
          RGB_feature3 = self.relu_RU_end3_img(RGB_feature3)
          # RU Depth
          Depth_feature = d3
          Depth_feature = self.relu_RU3_depth(Depth_feature)
          Depth_feature = self.relu_RU3_1_depth(self.bn_RU3_1_depth(self.conv_RU3_1_depth(Depth_feature)))
          Depth_feature = self.bn_RU3_2_depth(self.conv_RU3_2_depth(Depth_feature))
          Depth_feature3 = d3 + Depth_feature
          Depth_feature3 = self.relu_RU_end3_depth(Depth_feature3)
          # out
          RGB_feature3 = torch.mul(gate_out3,RGB_feature3)
          Depth_feature3 = torch.mul(gate_out3, Depth_feature3)
          feature3 = torch.cat((RGB_feature3,Depth_feature3),1)  #torch.Size([10, 512, 64, 64])
          feature3 = F.interpolate(feature3, size=[32, 32], mode="bilinear", align_corners=True)  #torch.Size([10, 512, 32, 32])

          ##### GSFM for Conv4_4 #####
          #concat h4 and d4
          img_feature4 = self.img_con2(h4)
          depth_feature4 = self.depth_con2(d4)
          cat_feature4 = torch.cat((img_feature4,depth_feature4),1)
          #Adaptation layer
          cat_feature4 = self.bn2(self.gate_rear2(cat_feature4))  #torch.Size([10, 512, 32, 32])
          feature4 = cat_feature4 + feature3
          #Integration layer
          feature4 = self.relu4_4(feature4)
          feature4 = self.sconv_relu4_4(self.sconv_bn4_4(self.sconv4_4(feature4)))  #torch.Size([10, 512, 32, 32])
          feature4 = F.interpolate(feature4, size=[16, 16], mode="bilinear", align_corners=True)  #torch.Size([10, 512, 16, 16])
          
          ##### GSFM for Conv5_4 #####
          #concat h5 and d5
          img_feature5 = self.img_con1(h5)
          depth_feature5 = self.depth_con1(d5)
          cat_feature5 = torch.cat((img_feature5,depth_feature5),1)
          #Adaptation layer
          cat_feature5 = self.bn5_4(self.gate_rear5_4(cat_feature5))  #torch.Size([10, 512, 16, 16])
          feature5 = cat_feature5 + feature4
          #Integration layer
          feature5 = self.relu5_4(feature5)
          out_depthmpt = self.sconv_relu5_4(self.sconv_bn5_4(self.sconv5_4(feature5)))  #torch.Size([10, 512, 16, 16])
          return out_depthmpt
