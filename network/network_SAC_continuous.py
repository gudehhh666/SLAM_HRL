import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
from PPO.model_resnet18_fuse import ResNet, ResBlock, ResNet_RGB
import torch.nn.init
from PPO.att import AttentionBlock
# from torchsummary import summary
from torch.distributions import Normal

device0 = torch.device("cuda:0")




class Actor(nn.Module):
     def __init__(self,
                    action_dim, droprate=0.5, feat_dim=1024):
          super(Actor, self).__init__()
          # self.max_action = max_action
          # self.max_action = torch.tensor([0.5, 0.3]).to(device0)  #有后退
          # self.max_action = torch.tensor([0.3, 0.2]).to(device0)  
          # self.max_action = torch.tensor([0.25, 0.3]).to(device0)   #无后退
          self.action_dim = action_dim
          self.droprate = droprate
          self.feature_extractor_depth = ResNet(ResBlock).to(device0)
          self.lin1 = nn.Linear(1, 256)
          self.lin2 = nn.Linear(1, 256)
          self.fc = nn.Linear(feat_dim, 128)
          self.fc1 = nn.Linear(128, 128)
          self.mean_layer = nn.Linear(128, action_dim)
          self.log_std_layer = nn.Linear(128, action_dim)

     def forward(self, depth, local_goal_point, angle):
          obv_s = self.feature_extractor_depth(depth)
          local_goal_point = local_goal_point.float()
          local_goal_point = local_goal_point.view(-1, 1).to(device0)
          local_goal_point_emb1 = F.leaky_relu(self.lin1(local_goal_point))
          angle = angle.view(-1, 1)
          angle_emb1 = F.leaky_relu(self.lin2(angle))
          scalar_values = torch.cat((local_goal_point_emb1, angle_emb1), 1)
          x = torch.cat((obv_s, scalar_values), 1)
          x = x.view(x.size(0), -1)
          # if self.droprate > 0:
          #      x = F.dropout(x, p=self.droprate)
          
          # print ('x.shape: ', x.shape)
          x = F.relu(self.fc(x))
          x = F.relu(self.fc1(x))
          mean = self.mean_layer(x)
          log_std = self.log_std_layer(x)
          log_std = torch.clamp(log_std, -20, 2.5)
          return mean, log_std.exp()

     def sample_normal(self, depth, local_goal_point, angle, reparameterize=True,
                                                                                epsilon=1e-6):
          mean, std = self.forward(depth, local_goal_point, angle)
          dist = Normal(mean, std)

          if reparameterize:
               # equals
               # e=distributions.Normal(0, 1).sample()
               # action = mean + e * std
               actions = dist.rsample()
          else:
               actions = dist.sample()
          
          action = torch.tanh(actions)
          log_probs = dist.log_prob(actions) - torch.log(1.-action.pow(2)+epsilon)
          log_probs = log_probs.sum(1, keepdim=True)
          return action, log_probs




class Critic(nn.Module):
     def __init__(self, droprate=0.5, feat_dim=1024):
          super(Critic, self).__init__()
          self.droprate = droprate
          self.feature_extractor_depth = ResNet(ResBlock).to(device0)
          self.lin1 = nn.Linear(1, 256)
          self.lin2 = nn.Linear(1, 256)
          self.fc = nn.Linear(feat_dim, 128)

          self.fc_action1 = nn.Linear(2, 256)

          # Q1
          self.l1 = nn.Linear(feat_dim + 256, 128)
          self.l2 = nn.Linear(128, 64)
          self.l3 = nn.Linear(64, 1)
          # # Q2
          # self.l4 = nn.Linear(feat_dim + 256, 128)
          # self.l5 = nn.Linear(128, 64)
          # self.l6 = nn.Linear(64, 1)

     def forward(self, depth, local_goal_point, angle, a):
          # print ('batch_t[img_next][index].shape: ', batch_t['img_next'][index].shape)
          obv_s = self.feature_extractor_depth(depth)
          local_goal_point = local_goal_point.float()
          local_goal_point = local_goal_point.view(-1, 1).to(device0)
          local_goal_point_emb1 = F.leaky_relu(self.lin1(local_goal_point))
          angle = angle.view(-1, 1)
          angle_emb1 = F.leaky_relu(self.lin2(angle))
          scalar_values = torch.cat((local_goal_point_emb1, angle_emb1), 1)
          x = torch.cat((obv_s, scalar_values), 1)
          x = x.view(x.size(0), -1)

          action_emb1 = F.leaky_relu(self.fc_action1(a))
          s_a = torch.cat((x, action_emb1), 1)
          s_a = s_a.view(s_a.size(0), -1)

          # if self.droprate > 0:
          #      x = F.dropout(x, p=self.droprate)
          

          q1 = F.relu(self.l1(s_a))
          q1 = F.relu(self.l2(q1))
          q1 = self.l3(q1)

          # q2 = F.relu(self.l4(s_a))
          # q2 = F.relu(self.l5(q2))
          # q2 = self.l6(q2)

          return q1



