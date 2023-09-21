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




class SAC_actor(nn.Module):
     def __init__(self,
                    action_dim, droprate=0.5, feat_dim=1024):
          super(SAC_actor, self).__init__()
          # self.max_action = max_action
          # self.max_action = torch.tensor([0.5, 0.3]).to(device0)  #有后退
          # self.max_action = torch.tensor([0.3, 0.2]).to(device0)  
          self.max_action = torch.tensor([0.25, 0.3]).to(device0)   #无后退
          self.action_dim = action_dim
          self.droprate = droprate
          self.feature_extractor_depth = ResNet(ResBlock).to(device0)
          self.lin1 = nn.Linear(1, 256)
          self.lin2 = nn.Linear(1, 256)
          self.fc = nn.Linear(feat_dim, 128)
          self.mean_layer = nn.Linear(128, action_dim)
          self.log_std_layer = nn.Linear(128, action_dim)

     def forward(self, depth, local_goal_point, angle, deterministic=False, with_logprob=True):
          obv_s = self.feature_extractor_depth(depth)
          local_goal_point = local_goal_point.float()
          local_goal_point = local_goal_point.view(-1, 1).to(device0)
          local_goal_point_emb1 = F.leaky_relu(self.lin1(local_goal_point))
          angle = angle.view(-1, 1)
          angle_emb1 = F.leaky_relu(self.lin2(angle))
          scalar_values = torch.cat((local_goal_point_emb1, angle_emb1), 1)
          x = torch.cat((obv_s, scalar_values), 1)
          x = x.view(x.size(0), -1)
          if self.droprate > 0:
               x = F.dropout(x, p=self.droprate)
          
          # print ('x.shape: ', x.shape)
          x = self.fc(x)
          mean = self.mean_layer(x)
          log_std = self.log_std_layer(x)
          log_std = torch.clamp(log_std, -20, 2)
          std = torch.exp(log_std)

          dist = Normal(mean, std)
          if deterministic:  # When evaluating，we use the deterministic policy
               a = mean
          else:
               a = dist.rsample()  # reparameterization trick: mean+std*N(0,1)
          
          if with_logprob:  # The method refers to Open AI Spinning up, which is more stable.
               # # print ('a.shape: ', a.shape)   #torch.Size([4, 2])
               # log_pi_test = dist.log_prob(a)
               # # print ('log_pi_test.shape: ', log_pi_test.shape)  #torch.Size([4, 2])
               log_pi = dist.log_prob(a).sum(dim=1, keepdim=True)
               # print ('log_pi.shape: ', log_pi.shape)   #torch.Size([4, 1])
               log_pi -= (2 * (np.log(2) - a - F.softplus(-2 * a))).sum(dim=1, keepdim=True)
          else:
               log_pi = None

          #===有后退===
          a = self.max_action * torch.tanh(a)

          # #===没有后退===
          # a[:,0] = (-0.5) * torch.sigmoid(a[:,0])
          # a[:,1] = 0.3 * torch.tanh(a[:,1])

          # print ('a.shape in SafeOptionNet() 2: ', a.shape)  #torch.Size([1, 2])

          return a, log_pi




class SAC_Critic(nn.Module):
     def __init__(self, droprate=0.5, feat_dim=1024):
          super(SAC_Critic, self).__init__()
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
          # Q2
          self.l4 = nn.Linear(feat_dim + 256, 128)
          self.l5 = nn.Linear(128, 64)
          self.l6 = nn.Linear(64, 1)

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

          if self.droprate > 0:
               x = F.dropout(x, p=self.droprate)
          

          q1 = F.relu(self.l1(s_a))
          q1 = F.relu(self.l2(q1))
          q1 = self.l3(q1)

          q2 = F.relu(self.l4(s_a))
          q2 = F.relu(self.l5(q2))
          q2 = self.l6(q2)

          return q1, q2



