import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
from PPO.model_resnet18_fuse import ResNet, ResBlock, ResNet_RGB
# from model_resnet18_fuse import ResNet, ResBlock, ResNet_RGB
import torch.nn.init
# from PPO.att import AttentionBlock
# from torchsummary import summary
from torch.distributions import Normal

device0 = torch.device("cuda:0")




class Actor(nn.Module):
     def __init__(self,
                    action_dim, droprate=0.5, feat_dim=1024):
          super(Actor, self).__init__()
          # self.max_action = max_action
          self.action_bound = torch.tensor([0.5, 0.3]).to(device0)  #有后退
          # self.max_action = torch.tensor([0.3, 0.2]).to(device0)  
          # self.action_bound = torch.tensor([0.25, 0.3]).to(device0)   #无后退
          self.action_dim = action_dim
          self.droprate = droprate
          self.feature_extractor_depth = ResNet(ResBlock).to(device0)
          self.lin1 = nn.Linear(1, 256)
          self.lin2 = nn.Linear(1, 256)
          self.fc = nn.Linear(feat_dim, 128)
          self.fc1 = nn.Linear(128, 128)

          self.l3 = nn.Linear(128, action_dim)



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
          a = self.action_bound * torch.tanh(self.l3(x))  # [-max,max]

          return a





class Critic(nn.Module):
     def __init__(self, droprate=0.5, feat_dim=1024):
          super(Critic, self).__init__()
          self.droprate = droprate

          # Q1
          self.Q1_feature_extractor_depth = ResNet(ResBlock).to(device0)
          self.Q1_lin1 = nn.Linear(1, 256)
          self.Q1_lin2 = nn.Linear(1, 256)
          self.Q1_fc = nn.Linear(feat_dim, 128)
          self.Q1_fc_action1 = nn.Linear(2, 256)
          self.Q1_l1 = nn.Linear(feat_dim + 256, 128)
          self.Q1_l2 = nn.Linear(128, 64)
          self.Q1_l3 = nn.Linear(64, 1)


          # Q2
          self.Q2_feature_extractor_depth = ResNet(ResBlock).to(device0)
          self.Q2_lin1 = nn.Linear(1, 256)
          self.Q2_lin2 = nn.Linear(1, 256)
          self.Q2_fc = nn.Linear(feat_dim, 128)
          self.Q2_fc_action1 = nn.Linear(2, 256)
          self.Q2_l1 = nn.Linear(feat_dim + 256, 128)
          self.Q2_l2 = nn.Linear(128, 64)
          self.Q2_l3 = nn.Linear(64, 1)

     
     def forward(self, depth, local_goal_point, angle, a):
          # print ('batch_t[img_next][index].shape: ', batch_t['img_next'][index].shape)

          net1_obv_s = self.Q1_feature_extractor_depth(depth)
          net1_local_goal_point = local_goal_point.float()
          net1_local_goal_point = net1_local_goal_point.view(-1, 1).to(device0)
          net1_local_goal_point_emb1 = F.leaky_relu(self.Q1_lin1(net1_local_goal_point))
          net1_angle = angle.view(-1, 1)
          net1_angle_emb1 = F.leaky_relu(self.Q1_lin2(net1_angle))
          net1_scalar_values = torch.cat((net1_local_goal_point_emb1, net1_angle_emb1), 1)
          net1_x = torch.cat((net1_obv_s, net1_scalar_values), 1)
          net1_x = net1_x.view(net1_x.size(0), -1)
          
          # print(a.shape)
          net1_action_emb1 = F.leaky_relu(self.Q1_fc_action1(a))
          net1_s_a = torch.cat((net1_x, net1_action_emb1), 1)
          net1_s_a = net1_s_a.view(net1_s_a.size(0), -1)
          q1 = F.relu(self.Q1_l1(net1_s_a))
          q1 = F.relu(self.Q1_l2(q1))
          q1 = self.Q1_l3(q1)

          net2_obv_s = self.Q2_feature_extractor_depth(depth)
          net2_local_goal_point = local_goal_point.float()
          net2_local_goal_point = net2_local_goal_point.view(-1, 1).to(device0)
          net2_local_goal_point_emb1 = F.leaky_relu(self.Q2_lin1(net2_local_goal_point))
          net2_angle = angle.view(-1, 1)
          net2_angle_emb1 = F.leaky_relu(self.Q2_lin2(net2_angle))
          net2_scalar_values = torch.cat((net2_local_goal_point_emb1, net2_angle_emb1), 1)
          net2_x = torch.cat((net2_obv_s, net2_scalar_values), 1)
          net2_x = net2_x.view(net2_x.size(0), -1)
          net2_action_emb1 = F.leaky_relu(self.Q2_fc_action1(a))
          net2_s_a = torch.cat((net2_x, net2_action_emb1), 1)
          net2_s_a = net2_s_a.view(net2_s_a.size(0), -1)
          q2 = F.relu(self.Q2_l1(net2_s_a))
          q2 = F.relu(self.Q2_l2(q2))
          q2 = self.Q2_l3(q2)

          return q1, q2
     
     def Q1(self, depth, local_goal_point, angle, a):
          net1_obv_s = self.Q1_feature_extractor_depth(depth)
          net1_local_goal_point = local_goal_point.float()
          net1_local_goal_point = net1_local_goal_point.view(-1, 1).to(device0)
          net1_local_goal_point_emb1 = F.leaky_relu(self.Q1_lin1(net1_local_goal_point))
          net1_angle = angle.view(-1, 1)
          net1_angle_emb1 = F.leaky_relu(self.Q1_lin2(net1_angle))
          net1_scalar_values = torch.cat((net1_local_goal_point_emb1, net1_angle_emb1), 1)
          net1_x = torch.cat((net1_obv_s, net1_scalar_values), 1)
          net1_x = net1_x.view(net1_x.size(0), -1)
          net1_action_emb1 = F.leaky_relu(self.Q1_fc_action1(a))
          net1_s_a = torch.cat((net1_x, net1_action_emb1), 1)
          net1_s_a = net1_s_a.view(net1_s_a.size(0), -1)
          q1 = F.relu(self.Q1_l1(net1_s_a))
          q1 = F.relu(self.Q1_l2(q1))
          q1 = self.Q1_l3(q1)
          return q1


class Option(nn.Module):
     def __init__(self, option_num = 3,droprate=0.5, feat_dim =1024):
          super(Option, self).__init__()
          self.droprate = droprate
          self.option_num = option_num
          
          
          # Q1
          self.feature_extractor_depth = ResNet(ResBlock).to(device0)
          self.lin1 = nn.Linear(1, 256)
          self.lin2 = nn.Linear(1, 256)
          self.fc = nn.Linear(feat_dim, 128)
          self.fc_action1 = nn.Linear(2, 256)
          
          self.l1 = nn.Linear(feat_dim + 256, 128)
          self.l2 = nn.Linear(128, 64)
          self.l3 = nn.Linear(64, self.option_num)
          
          self.dec1 = nn.Linear(self.option_num, 128)
          self.dec2 = nn.Linear(128, 64)
          self.dec3 = nn.Linear(64, feat_dim + 256)
          
     def forward(self, depth, local_goal_point, angle, a):
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
          
          s_a_out = torch.relu(self.l1(s_a))
          s_a_out = torch.relu(self.l2(s_a_out))
          s_a_out = self.l3(s_a_out)
          
          o_dec = torch.relu(self.dec1(s_a_out))
          o_dec = torch.relu(self.dec2(o_dec))
          out_dec = self.dec3(o_dec)
          
          
          
          out_option = torch.softmax(s_a_out, dim=1)
          
          return out_option, s_a, out_dec
          
          
class HRLTD3(nn.Module):
     def __init__(self, action_dim, option_num = 3,droprate=0.5, feat_dim =1024):
          super(HRLTD3, self).__init__()
          self.droprate = droprate
          self.option_num = option_num
          self.action_dim = action_dim
          self.actor_list = nn.ModuleList(Actor(self.action_dim).to(device0) for actor in range(option_num))
          self.critic = Critic().to(device0)
          self.option = Option(self.option_num).to(device0)
          
          
     # def softmax_option_target(self, depth, local_goal_point, angle, a):
          
     #      Q_predict = []
     #      n = depth.shape[0]
     #      for o in range(self.option_num):
     #           action_i = self.predict_actor_target(depth, local_goal_point, angle, o)
     #           Q_predict_i, _ = self.predict_critic_target(depth, local_goal_point, angle, action_i)
               
     #           if o == 0 :
     #                Q_predict = Q_predict_i.view(-1, 1)
     #           else:
     #                Q_predict = torch.cat((Q_predict, Q_predict_i.view(-1, 1)), dim = 1)
                    
     #      p = torch.softmax(Q_predict, dim = 1)
     #      p = p.numpy()
     #      row, col = p.shape


     #      p_sum = np.reshape(np.sum(p, axis=1), (row, 1))
     #      print('p_sum.shape: ', p_sum.shape)
     #      p_normalized = p  / np.matlib.repmat(p_sum, 1, col)
     #      print('p_normalized.shape: ', p_normalized.shape)
     #      p_cumsum = np.matrix(np.cumsum( p_normalized, axis=1))
     #      # print(p_cumsum[0])
     #      rand = np.matlib.repmat(np.random.random((row, 1)), 1, col)
     #      # print(rand[0])
     #      o_softmax = np.argmax(p_cumsum >= rand, axis=1)
          
     
          
     #      n = Q_predict.shape[0]
          
     #      Q_softmax = Q_predict[np.arange(n), o_softmax.flatten()]
          
     #      return Q_softmax, Q_softmax.view(n, 1), Q_predict
                    

               
               
     
     
     # def predict_critic_target(self, depth, local_goal_point, angle, a):
     #      return self.critic(depth, local_goal_point, angle, a)
          
     
     
     
     # def predict_actor_target(self, depth, local_goal_point, angle, option):
     #      action_list = []
     #      for actor in self.actors:
     #           action_o = actor(depth, local_goal_point, angle)
     #           action_list.append(action_o)
          
     #      action_list = torch.cat(action_list, 1)
     #      n = depth.shape[0]
     #      # n is the batch size actually
          
     #      if n ==1 or np.isscalar(option):
     #           action =action_list[option]
     #      else:
     #           for i in range(n):
     #                action = action_list[option[i]][i, :]
     #           else:
     #                action = torch.vstack(action, action_list[option[i]][i, :])
     #                # action = np.vstack(action, action_list[option[i]][i, :])
     #      return action          
          
          
          

          
          
          
          

          
          