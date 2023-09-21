import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
from PPO.model_resnet18_fuse import ResNet, ResBlock

device0 = torch.device("cuda:0")


class Actor_Net(nn.Module):
     def __init__(self, out_dim):
          super(Actor_Net, self).__init__()
          self.out_dim = out_dim
          self.model = ResNet(ResBlock).to(device0)
          self.lin1 = nn.Linear(1, 256)
          self.lin2 = nn.Linear(1, 256)
          self.lin3 = nn.Linear(1024, 128)
          self.lin4 = nn.Linear(128, 64)
          self.lin5 = nn.Linear(64, self.out_dim)
          self.lin6 = nn.Softmax(dim=1) #注意是沿着那个维度计算
     
     def forward(self, img, depth, mpt, local_goal_point, angle):
          # print ('img.shape: ', img.shape)
          print ('depth.shape: ', depth.shape)
          # print ('mpt.shape: ', mpt.shape)
          # obv_s = torch.cat((img, depth, mpt), 1)
          obv_s = depth
          output1 = self.model(obv_s)
          local_goal_point = local_goal_point.float()
          local_goal_point = local_goal_point.view(-1, 1).to(device0)
          local_goal_point_emb1 = F.leaky_relu(self.lin1(local_goal_point))
          angle = angle.view(-1, 1)
          angle_emb1 = F.leaky_relu(self.lin2(angle))
          scalar_values = torch.cat((local_goal_point_emb1, angle_emb1), 1)
          fuse_values = torch.cat((output1, scalar_values), 1)
          fuse_emb1 = F.leaky_relu(self.lin3(fuse_values))
          fuse_emb2 = F.leaky_relu(self.lin4(fuse_emb1))
          fuse_emb3 = self.lin5(fuse_emb2)
          out = self.lin6(fuse_emb3)
          return out


class Critic_Net(nn.Module):
     def __init__(self, out_dim):
          super(Critic_Net, self).__init__()
          self.out_dim = out_dim
          self.model = ResNet(ResBlock).to(device0)
          self.lin1 = nn.Linear(1, 256)
          self.lin2 = nn.Linear(1, 256)
          self.lin3 = nn.Linear(1024, 128)
          self.lin4 = nn.Linear(128, 64)
          self.lin5 = nn.Linear(64, self.out_dim)
          self.lin6 = nn.Softmax(dim=1) #注意是沿着那个维度计算
     
     def forward(self, img, depth, mpt, local_goal_point, angle):
          # obv_s = torch.cat((img, depth, mpt), 1)
          obv_s = depth
          output1 = self.model(obv_s)
          local_goal_point = local_goal_point.float()
          local_goal_point = local_goal_point.view(-1, 1).to(device0)
          local_goal_point_emb1 = F.leaky_relu(self.lin1(local_goal_point))
          angle = angle.view(-1, 1)
          angle_emb1 = F.leaky_relu(self.lin2(angle))
          scalar_values = torch.cat((local_goal_point_emb1, angle_emb1), 1)
          fuse_values = torch.cat((output1, scalar_values), 1)
          fuse_emb1 = F.leaky_relu(self.lin3(fuse_values))
          fuse_emb2 = F.leaky_relu(self.lin4(fuse_emb1))
          fuse_emb3 = self.lin5(fuse_emb2)
          return fuse_emb3




