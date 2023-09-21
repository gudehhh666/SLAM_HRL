import torch
import torch.nn as nn
import numpy as np
# from model import RGBNet,DepthNet
from torch.autograd import Variable
from model import RGBNet,DepthNet,MptNet
from my_fusion import Fusion_depthmpt,Fusion_rgbmpt,Fusion_rgbdepth
from triplet import TripletAttention_new
from torchsummary import summary
from pytorch_model_summary import summary as summary2
import torch.nn.functional as F
import time

device0 = torch.device("cuda:0")

class Net(nn.Module):
     def __init__(self, out_dim):
          super(Net, self).__init__()
          self.out_dim = out_dim
          self.model_rgb = RGBNet().cuda()
          self.model_depth = DepthNet().cuda()
          self.model_mpt = MptNet().cuda()
          self.fuse_rgbdepth = Fusion_rgbdepth().cuda()
          self.fuse_rgbmpt = Fusion_rgbmpt().cuda()
          self.fuse_depthmpt = Fusion_depthmpt().cuda()
          self.triplet_attention = TripletAttention_new().cuda()
          self.lin1 = nn.Linear(3, 512)
          # self.vector_m = nn.AdaptiveAvgPool2d((2, 2))
          self.vector_m = nn.AdaptiveAvgPool2d((1,1))
          self.lin2 = nn.Linear(1024, self.out_dim)
          self.lin3 = nn.Softmax(dim=0) #注意是沿着那个维度计算

     def forward(self, img, depth, mpt, local_goal_point, angle):
          cnt1 = time.time()
          h1,h2,h3,h4,h5 = self.model_rgb(img)       # RGBNet's output
          print ('model_rgb time: ', time.time() - cnt1)
          cnt2 = time.time()
          d1,d2,d3,d4,d5 = self.model_depth(depth)    # DepthNet's output
          print ('model_depth time: ', time.time() - cnt2)
          cnt3 = time.time()
          m1,m2,m3,m4,m5 = self.model_mpt(mpt)    # MptNet's output
          print ('model_mpt time: ', time.time() - cnt3)
          cnt4 = time.time()
          output_rgbdepth = self.fuse_rgbdepth(h1,h2,h3,h4,h5,d1,d2,d3,d4,d5)     # Final output
          print ('fuse_rgbdepth time: ', time.time() - cnt4)
          cnt5 = time.time()
          output_rgbmpt = self.fuse_rgbmpt(h1,h2,h3,h4,h5,m1,m2,m3,m4,m5)
          print ('fuse_rgbmpt time: ', time.time() - cnt5)
          cnt6 = time.time()
          output_depthmpt = self.fuse_depthmpt(d1,d2,d3,d4,d5,m1,m2,m3,m4,m5)
          print ('fuse_depthmpt time: ', time.time() - cnt6)
          cnt7 = time.time()
          output_fuse = self.triplet_attention(output_rgbdepth, output_rgbmpt, output_depthmpt)
          print ('triplet_attention time: ', time.time() - cnt7)
          # output = output.view(output.size(0), -1)
          # output = F.relu(self.lin1(output))
          output_fuse = self.vector_m(output_fuse)
          output_fuse = output_fuse.view(output_fuse.size(0), -1)
          # local_goal_point = local_goal_point[:,0,:]
          local_goal_point = local_goal_point.float()
          local_goal_point = local_goal_point.view(-1, 2).to(device0)
          # local_goal_point = torch.tensor(local_goal_point).to(device0)
          # angle = angle[:,0,:]
          angle = angle.view(-1, 1)
          # print ('local_goal_point.shape: ', local_goal_point.shape)
          # print ('angle.shape: ', angle.shape)
          # print ('local_goal_point.dtype: ', local_goal_point.dtype)
          # print ('angle.dtype: ', angle.dtype)
          scalar_values = torch.cat((local_goal_point, angle), 1)
          # print ('scalar_values.dtype: ', scalar_values.dtype)
          scalar_values = self.lin1(scalar_values)
          cat = F.relu(torch.cat((output_fuse, scalar_values), 1))
          final = F.relu(self.lin2(cat))
          out = self.lin3(final)
          return out


if __name__ == "__main__":
     device0 = torch.device("cuda:0")
     mean_rgb = np.array([0.447, 0.407, 0.386])
     std_rgb = np.array([0.244, 0.250, 0.253])

     img = np.random.rand(3, 3, 256, 256)
     img = img.astype(np.float32)/255.0
     img[:, 0, :, :] -= mean_rgb[0]
     img[:, 1, :, :] -= mean_rgb[1]
     img[:, 2, :, :] -= mean_rgb[2]
     img[:, 0, :, :] /= std_rgb[0]
     img[:, 1, :, :] /= std_rgb[1]
     img[:, 2, :, :] /= std_rgb[2]

     # num = img.shape[0]
     # for i in range(num):
     #      img[i] -=  mean_rgb
     #      img[i] /= std_rgb
     # img -= mean_rgb
     # img /= std_rgb
     img = torch.from_numpy(img).float()

     depth = np.random.rand(3, 256, 256, 1)
     depth = depth.astype(np.float32)/255.0
     depth = torch.from_numpy(depth).float()

     mpt = np.random.rand(3, 256, 256, 1)
     mpt = mpt.astype(np.float32)/255.0
     mpt = torch.from_numpy(mpt).float()

     img, mpt, depth = Variable(img), Variable(mpt), Variable(depth)
     n, c, h, w = img.size()        # batch_size, channels, height, weight
     depth = depth.view(n,h,w,1).repeat(1,1,1,c)
     depth = depth.transpose(3,1)
     depth = depth.transpose(3,2)
     mpt = mpt.view(n,h,w,1).repeat(1,1,1,c)
     mpt = mpt.transpose(3,1)
     mpt = mpt.transpose(3,2)
     img = img.to(device0)
     depth = depth.to(device0)
     mpt = mpt.to(device0)

     local_goal_point = np.random.rand(3, 2)
     angle = np.random.rand(3, 1)
     local_goal_point = torch.from_numpy(local_goal_point).float()
     angle = torch.from_numpy(angle).float()
     local_goal_point = local_goal_point.to(device0)
     angle = angle.to(device0)

     network =  Net(3).cuda()
     # cnt = time.time()
     output = network(img, depth, mpt, local_goal_point, angle)
     # print ('total time: ', time.time() - cnt)
     # print (output.shape)

     #from torchsummary import summary
     summary(network, input_size=[(3, 256, 256), (3, 256, 256), (3, 256, 256), (1, 2), (1, 1)], batch_size = 1, device='cuda')
     
     # #pytorch_model_summary
     # img =  torch.zeros((1, 3, 256, 256)).to(device0)
     # depth = torch.zeros((1, 3, 256, 256)).to(device0)
     # mpt = torch.zeros((1, 3, 256, 256)).to(device0)
     # local_goal_point = torch.zeros((1, 2)).to(device0)
     # angle = torch.zeros((1, 1)).to(device0)
     # print (summary2(network, img, depth, mpt, local_goal_point, angle, show_input=False, show_hierarchical=True))
