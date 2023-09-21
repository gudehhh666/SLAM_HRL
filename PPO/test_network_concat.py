import torch
import torch.nn as nn
import numpy as np
# from model import RGBNet,DepthNet
from torch.autograd import Variable
from model_resnet18 import ResNet18Image,ResNet18Depth,ResNet18Mpt
# from model_resnet18 import ResNet18Image,ResNet18Depth,ResNet18Mpt
# from PPO.my_fusion import Fusion_depthmpt,Fusion_rgbmpt,Fusion_rgbdepth
# from PPO.triplet import TripletAttention_new
from torchsummary import summary
import torch.nn.functional as F
import time
from torch.distributions import MultivariateNormal
from torch.optim import Adam

device0 = torch.device("cuda:0")


class Actor_Net(nn.Module):
     def __init__(self, out_dim):
          super(Actor_Net, self).__init__()
          self.out_dim = out_dim

          self.model_rgb = ResNet18Image().to(device0)
          self.model_depth = ResNet18Depth().to(device0)
          self.model_mpt = ResNet18Mpt().to(device0)
          self.lin1 = nn.Linear(3, 1536)
          self.lin2 = nn.Linear(3072, self.out_dim)
          self.lin3 = nn.Softmax(dim=1) #注意是沿着那个维度计算


     def forward(self, img, depth, mpt, local_goal_point, angle):
          rgb_feature = self.model_rgb(img)
          depth_feature = self.model_depth(depth)
          mpt_feature = self.model_mpt(mpt)
          output_fuse = torch.cat((rgb_feature, depth_feature, mpt_feature), 1)
          local_goal_point = local_goal_point.float()
          local_goal_point = local_goal_point.view(-1, 2).to(device0)
          angle = angle.view(-1, 1)
          scalar_values = torch.cat((local_goal_point, angle), 1)
          print ('scalar_values: ', scalar_values)
          scalar_values = F.leaky_relu(self.lin1(scalar_values))
          # cat = F.relu(torch.cat((output_fuse, scalar_values), 1))
          # final = F.relu(self.lin2(cat))
          # cat = F.leaky_relu(torch.cat((output_fuse, scalar_values), 1))
          # final = F.leaky_relu(self.lin2(cat))
          cat = F.leaky_relu(torch.cat((output_fuse, scalar_values), 1))
          cat_2 = self.lin2(cat)
          print ('cat_2: ', cat_2)
          print (self.lin2.weight)
          print (self.lin2.weight.grad)
          # final = torch.sigmoid(cat_2)
          # print ('final: ', final)
          final = F.leaky_relu(cat_2)
          print ('final: ', final)
          out = self.lin3(final)
          return out


class Critic_Net(nn.Module):
     def __init__(self, out_dim):
          super(Critic_Net, self).__init__()
          self.out_dim = out_dim

          self.model_rgb = ResNet18Image().to(device0)
          self.model_depth = ResNet18Depth().to(device0)
          self.model_mpt = ResNet18Mpt().to(device0)
          self.lin1 = nn.Linear(3, 1536)
          self.lin2 = nn.Linear(3072, self.out_dim)
          # self.lin3 = nn.Softmax(dim=0) #注意是沿着那个维度计算


     def forward(self, img, depth, mpt, local_goal_point, angle):
          rgb_feature = self.model_rgb(img)
          depth_feature = self.model_depth(depth)
          mpt_feature = self.model_mpt(mpt)
          output_fuse = torch.cat((rgb_feature, depth_feature, mpt_feature), 1)
          local_goal_point = local_goal_point.float()
          local_goal_point = local_goal_point.view(-1, 2).to(device0)
          angle = angle.view(-1, 1)
          scalar_values = torch.cat((local_goal_point, angle), 1)
          scalar_values = F.leaky_relu(self.lin1(scalar_values))
          # cat = F.relu(torch.cat((output_fuse, scalar_values), 1))
          # cat = F.leaky_relu(torch.cat((output_fuse, scalar_values), 1))
          cat = F.leaky_relu(torch.cat((output_fuse, scalar_values), 1))
          # final = F.relu(self.lin2(cat))
          final = self.lin2(cat)
          # out = self.lin3(final)
          return final



if __name__ == "__main__":
     # device0 = torch.device("cuda:0")
     device1 = torch.device("cpu")

     # network =  ResNet18Image().to(device0)
     img = np.random.rand(1, 3, 256, 256)
     img = torch.from_numpy(img).float().to(device0)
     # output1 = network(img)
     # print (output1.shape)
     depth = np.random.rand(1, 3, 256, 256)
     depth = torch.from_numpy(depth).float().to(device0)

     mpt = np.random.rand(1, 3, 256, 256)
     mpt = torch.from_numpy(mpt).float().to(device0)

     local_goal_point = np.random.rand(1, 2)
     angle = np.random.rand(1, 1)
     local_goal_point = torch.from_numpy(local_goal_point).float()
     angle = torch.from_numpy(angle).float()
     local_goal_point = local_goal_point.to(device0)
     angle = angle.to(device0)

     network =  Actor_Net(3).cuda()
     # cnt = time.time()
     output = network(img, depth, mpt, local_goal_point, angle)
     print ('output: ', output)
     output = output.to(device1)

     # cov_var = torch.full(size=(3,), fill_value=0.5)
     # cov_mat = torch.diag(cov_var)
     # dist = MultivariateNormal(output, cov_mat)
     # action = dist.sample()
     # print ('action: ', action)

     # summary(network, input_size=[(3, 256, 256), (3, 256, 256), (3, 256, 256), (1, 2), (1, 1)], batch_size = 1, device='cuda')

     # # actor_optim = Adam(network.parameters())
     # # for x in actor_optim.param_groups[0]['params']:
     # #      print (x)

     print (network)
     print (network.model_rgb.resnet.conv1.weight)
     print (network.model_rgb.resnet.conv1.weight.grad)
     print (network.model_rgb.resnet.conv1.weight.shape)
     # print (network.lin2.weight)
     # print (network.lin2.weight.grad)







