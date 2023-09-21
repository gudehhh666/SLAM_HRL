import torch
import torch.nn as nn
import numpy as np
# from model import RGBNet,DepthNet
from torch.autograd import Variable
from PPO.model_resnet18 import ResNet18Image,ResNet18Depth,ResNet18Mpt
# from model_resnet18 import ResNet18Image,ResNet18Depth,ResNet18Mpt
# from PPO.my_fusion import Fusion_depthmpt,Fusion_rgbmpt,Fusion_rgbdepth
# from PPO.triplet import TripletAttention_new
from torchsummary import summary
import torch.nn.functional as F
import time
from torch.distributions import MultivariateNormal


device0 = torch.device("cuda:0")


class Actor_Net(nn.Module):
     def __init__(self, out_dim):
          super(Actor_Net, self).__init__()
          self.out_dim = out_dim

          # #input: img，depth, mpt , local_goal_point, angle (fuse:sum)
          # self.model_rgb = ResNet18Image().to(device0)
          # self.model_depth = ResNet18Depth().to(device0)
          # self.model_mpt = ResNet18Mpt().to(device0)
          # self.lin1 = nn.Linear(3, 1536)
          # self.lin2 = nn.Linear(1024, 2000)
          # self.lin3 = nn.Linear(1536, 10)
          # self.lin4 = nn.Linear(10, self.out_dim)
          # self.lin5 = nn.Softmax(dim=1) #注意是沿着那个维度计算

          #input: angle, local_goal_point
          self.lin1 = nn.Linear(1, 128)
          # nn.init.xavier_uniform_(self.lin1.weight)
          # self.lin1.bias.fill_(0.01)
          self.lin2 = nn.Linear(1, 128)
          # nn.init.xavier_uniform_(self.lin2.weight)
          # self.lin2.bias.fill_(0.01)
          self.lin3 = nn.Linear(256, 128)
          # nn.init.xavier_uniform_(self.lin3.weight)
          # self.lin3.bias.fill_(0.01)
          self.lin4 = nn.Linear(128, 64)
          # nn.init.xavier_uniform_(self.lin4.weight)
          # self.lin4.bias.fill_(0.01)
          self.lin5 = nn.Linear(64, self.out_dim)
          # nn.init.xavier_uniform_(self.lin5.weight)
          # self.lin5.bias.fill_(0.01)
          self.lin6 = nn.Softmax(dim=1) #注意是沿着那个维度计算

          # self.test_nan = torch.tensor([False, False,  False]).float().to(device0)



     # def forward(self, img, depth, mpt, local_goal_point, angle): #input: img，depth, mpt , local_goal_point (fuse:cat)
     #      rgb_feature = self.model_rgb(img)
     #      depth_feature = self.model_depth(depth)
     #      mpt_feature = self.model_mpt(mpt)
     #      output_fuse = torch.cat((rgb_feature, depth_feature, mpt_feature), 1)
     #      # output_fuse = torch.cat((rgb_feature, depth_feature), 1)
     #      local_goal_point = local_goal_point.float()
     #      local_goal_point = local_goal_point.view(-1, 2).to(device0)
     #      angle = angle.view(-1, 1)
     #      scalar_values = torch.cat((local_goal_point, angle), 1)
     #      # print ('scalar_values: ', scalar_values)
     #      scalar_values = F.leaky_relu(self.lin1(scalar_values))
     #      cat = F.relu(torch.cat((output_fuse, scalar_values), 1))
     #      final = F.relu(self.lin2(cat))
     #      # cat = F.leaky_relu(torch.cat((output_fuse, scalar_values), 1))
     #      # final = F.leaky_relu(self.lin2(cat))
     #      cat = F.leaky_relu(torch.cat((rgb_feature, scalar_values), 1))
     #      # cat = F.leaky_relu(torch.cat((output_fuse, scalar_values), 1))
     #      cat_2 = self.lin2(cat)
     #      print ('cat_2: ', cat_2)
          
     #      # final = torch.sigmoid(cat_2)
     #      # print ('final: ', final)
     #      final = F.leaky_relu(cat_2)
     #      print ('final: ', final)
     #      final_1 = F.leaky_relu(self.lin3(final))
     #      print ('final_1: ', final_1)
     #      # final_2 = torch.sigmoid(self.lin4(final_1))
     #      final_2 = F.leaky_relu(self.lin4(final_1))
     #      print ('final_2: ', final_2)
     #      out = self.lin5(final_2)
     #      print ('actor out: ', out)
     #      return out

     

     # def forward(self, img, depth, mpt, local_goal_point, angle):   #input: img
     #      rgb_feature = self.model_rgb(img)
     #      final_1 = F.leaky_relu(self.lin3(rgb_feature))
     #      print ('final_1: ', final_1)
     #      # final_2 = torch.sigmoid(self.lin4(final_1))
     #      final_2 = F.leaky_relu(self.lin4(final_1))
     #      print ('final_2: ', final_2)
     #      out = self.lin5(final_2)
     #      print ('actor out: ', out)
     #      return out
     
     # def forward(self, img, depth, mpt, local_goal_point, angle):   #input: img，depth
     #      rgb_feature = self.model_rgb(img)
     #      depth_feature = self.model_depth(depth)
     #      output_fuse = torch.cat((rgb_feature, depth_feature), 1)
     #      final_1 = F.leaky_relu(self.lin3(output_fuse))
     #      print ('final_1: ', final_1)
     #      # final_2 = torch.sigmoid(self.lin4(final_1))
     #      final_2 = F.leaky_relu(self.lin4(final_1))
     #      print ('final_2: ', final_2)
     #      out = self.lin5(final_2)
     #      print ('actor out: ', out)
     #      return out
     
     # def forward(self, img, depth, mpt, local_goal_point, angle):   #input: img，depth, mpt
     #      rgb_feature = self.model_rgb(img)
     #      depth_feature = self.model_depth(depth)
     #      mpt_feature = self.model_depth(mpt)
     #      output_fuse = torch.cat((rgb_feature, depth_feature, mpt_feature), 1)
     #      final_1 = F.leaky_relu(self.lin3(output_fuse))
     #      print ('final_1: ', final_1)
     #      # final_2 = torch.sigmoid(self.lin4(final_1))
     #      final_2 = F.leaky_relu(self.lin4(final_1))
     #      print ('final_2: ', final_2)
     #      out = self.lin5(final_2)
     #      print ('actor out: ', out)
     #      return out
     
     # def forward(self, img, depth, mpt, local_goal_point, angle):   #input: img，depth, mpt , local_goal_point, angle (fuse:sum)
     #      rgb_feature = self.model_rgb(img)
     #      depth_feature = self.model_depth(depth)
     #      mpt_feature = self.model_depth(mpt)
     #      output_fuse = torch.cat((rgb_feature, depth_feature, mpt_feature), 1)

     #      local_goal_point = local_goal_point.float()
     #      local_goal_point = local_goal_point.view(-1, 2).to(device0)
     #      angle = angle.view(-1, 1)
     #      scalar_values = torch.cat((local_goal_point, angle), 1)
     #      scalar_values2 = F.leaky_relu(self.lin1(scalar_values))

     #      scalar_values3 = output_fuse + scalar_values2

     #      final_1 = F.leaky_relu(self.lin3(scalar_values3))
     #      # print ('final_1: ', final_1)
     #      # final_2 = torch.sigmoid(self.lin4(final_1))
     #      final_2 = F.leaky_relu(self.lin4(final_1))
     #      # print ('final_2: ', final_2)
     #      out = self.lin5(final_2)
     #      print ('actor out: ', out)
     #      return out
     

     def forward(self, img, depth, mpt, local_goal_point, angle):   #input: local_goal_point/relative dist, angle
          local_goal_point = local_goal_point.float()
          # local_goal_point = local_goal_point / 20.0
          local_goal_point = local_goal_point.view(-1, 1).to(device0)
          # print ('Actor local_goal_point: ', local_goal_point)
          local_goal_point_emb1 = F.leaky_relu(self.lin1(local_goal_point))
          angle = angle.view(-1, 1)
          # print ('Actor angle: ', angle)
          angle_emb1 = F.leaky_relu(self.lin2(angle))
          scalar_values = torch.cat((local_goal_point_emb1, angle_emb1), 1)
          fuse_emb1 = F.leaky_relu(self.lin3(scalar_values))
          fuse_emb2 = F.leaky_relu(self.lin4(fuse_emb1))
          fuse_emb3 = self.lin5(fuse_emb2)
          out = self.lin6(fuse_emb3)
          # print ('actor out: ', out)
          x_1 = torch.isnan(out)
          x_1 = x_1.float()
          test_nan = torch.zeros_like(x_1).float().to(device0)
          nan = x_1.equal(test_nan)
          if (nan == False):
               print ('Error!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!')
               print ('local_goal_point_emb1: ', local_goal_point_emb1)
               print ('angle_emb1: ', angle_emb1)
               print ('scalar_values: ', scalar_values)
               print ('fuse_emb1: ', fuse_emb1)
               print ('fuse_emb2: ', fuse_emb2)
               print ('fuse_emb3: ', fuse_emb3)
               glovar.nan = 1

          return out







class Critic_Net(nn.Module):
     def __init__(self, out_dim):
          super(Critic_Net, self).__init__()
          self.out_dim = out_dim

          # #old
          # self.model_rgb = ResNet18Image().to(device0)
          # self.model_depth = ResNet18Depth().to(device0)
          # self.model_mpt = ResNet18Mpt().to(device0)
          # self.lin1 = nn.Linear(3, 1536)
          # self.lin2 = nn.Linear(3072, 2000)
          # self.lin3 = nn.Linear(2000, 1000)
          # self.lin4 = nn.Linear(1000, self.out_dim)
          # # self.lin2 = nn.Linear(3072, self.out_dim)
          # # self.lin3 = nn.Softmax(dim=0) #注意是沿着那个维度计算

          #input: local_goal_point
          # self.lin1 = nn.Linear(3, 10)
          # self.lin2 = nn.Linear(10, 5)
          # self.lin3 = nn.Linear(5, self.out_dim)

          #input: angle, local_goal_point
          self.lin1 = nn.Linear(1, 128)
          # nn.init.xavier_uniform_(self.lin1.weight)
          # self.lin1.bias.fill_(0.01)
          self.lin2 = nn.Linear(1, 128)
          # nn.init.xavier_uniform_(self.lin2.weight)
          # self.lin2.bias.fill_(0.01)
          self.lin3 = nn.Linear(256, 128)
          # nn.init.xavier_uniform_(self.lin3.weight)
          # self.lin3.bias.fill_(0.01)
          self.lin4 = nn.Linear(128, 64)
          # nn.init.xavier_uniform_(self.lin4.weight)
          # self.lin4.bias.fill_(0.01)
          self.lin5 = nn.Linear(64, self.out_dim)
          # nn.init.xavier_uniform_(self.lin5.weight)
          # self.lin5.bias.fill_(0.01)
          self.lin6 = nn.Softmax(dim=1) #注意是沿着那个维度计算



          #old
     # def forward(self, img, depth, mpt, local_goal_point, angle):
     #      rgb_feature = self.model_rgb(img)
     #      depth_feature = self.model_depth(depth)
     #      mpt_feature = self.model_mpt(mpt)
     #      output_fuse = torch.cat((rgb_feature, depth_feature, mpt_feature), 1)
     #      local_goal_point = local_goal_point.float()
     #      local_goal_point = local_goal_point.view(-1, 2).to(device0)
     #      angle = angle.view(-1, 1)
     #      scalar_values = torch.cat((local_goal_point, angle), 1)
     #      scalar_values = F.leaky_relu(self.lin1(scalar_values))
     #      # cat = F.relu(torch.cat((output_fuse, scalar_values), 1))
     #      # cat = F.leaky_relu(torch.cat((output_fuse, scalar_values), 1))
     #      cat = F.leaky_relu(torch.cat((output_fuse, scalar_values), 1))
     #      # final = F.relu(self.lin2(cat))
     #      final = self.lin2(cat)
     #      # out = self.lin3(final)
     #      # print ('critic final: ', final)
     #      final_1 = F.leaky_relu(self.lin3(final))
     #      # print ('critic final_1: ', final_1)
     #      final_2 = F.leaky_relu(self.lin4(final_1))
     #      # print ('critic final_2: ', final_2)
     #      return final_2
     

     def forward(self, img, depth, mpt, local_goal_point, angle):
          local_goal_point = local_goal_point.float()
          local_goal_point = local_goal_point.view(-1, 1).to(device0)
          local_goal_point_emb1 = F.leaky_relu(self.lin1(local_goal_point))
          angle = angle.view(-1, 1)
          angle_emb1 = F.leaky_relu(self.lin2(angle))
          scalar_values = torch.cat((local_goal_point_emb1, angle_emb1), 1)
          fuse_emb1 = F.leaky_relu(self.lin3(scalar_values))
          fuse_emb2 = F.leaky_relu(self.lin4(fuse_emb1))
          fuse_emb3 = self.lin5(fuse_emb2)
          return fuse_emb3



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

     cov_var = torch.full(size=(3,), fill_value=0.5)
     cov_mat = torch.diag(cov_var)
     dist = MultivariateNormal(output, cov_mat)
     action = dist.sample()
     print ('action: ', action)


