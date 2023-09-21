import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
from PPO.model_resnet18_fuse import ResNet, ResBlock, ResNet_RGB
import torch.nn.init
from PPO.att import AttentionBlock
from torchsummary import summary

device0 = torch.device("cuda:0")




class SafeOptionNet(nn.Module):
     def __init__(self,
                    action_dim, droprate=0.5, feat_dim=2048):
          super(SafeOptionNet, self).__init__()
          self.action_dim = action_dim
          self.droprate = droprate
          self.feature_extractor_RGB = ResNet(ResBlock).to(device0)
          self.feature_extractor_depth = ResNet(ResBlock).to(device0)
          self.feature_extractor_mpt = ResNet(ResBlock).to(device0)
          self.lin1 = nn.Linear(1, 256)
          self.lin2 = nn.Linear(1, 256)
          self.fc = nn.Linear(feat_dim, 128)
          self.fc_actor1 = nn.Linear(128, self.action_dim)
          self.lin6 = nn.Softmax(dim=1) #注意是沿着那个维度计算
          self.fc2 = nn.Linear(feat_dim, 128)
          self.fc_critic1 = nn.Linear(128, 1)
          self.std = nn.Parameter(torch.zeros((1, self.action_dim)))

     def forward(self, img, depth, mpt, local_goal_point, angle):
          img_emb = self.feature_extractor_RGB(img)
          depth_emb = self.feature_extractor_depth(depth)
          mpt_emb = self.feature_extractor_mpt(mpt)
          obv_s = torch.cat((img_emb, depth_emb, mpt_emb), 1)
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
          output = self.fc(x)
          output = self.fc_actor1(output)
          mean = self.lin6(output)
          std = F.softplus(self.std).expand(mean.size(0), -1)

          return {
               'mean': mean,
               'std': std,
          }



class CollideOptionNet(nn.Module):
     def __init__(self,
                    action_dim, droprate=0.5, feat_dim=2048):
          super(CollideOptionNet, self).__init__()
          self.action_dim = action_dim
          self.droprate = droprate
          self.feature_extractor_RGB = ResNet(ResBlock).to(device0)
          self.feature_extractor_depth = ResNet(ResBlock).to(device0)
          self.feature_extractor_mpt = ResNet(ResBlock).to(device0)
          self.lin1 = nn.Linear(1, 256)
          self.lin2 = nn.Linear(1, 256)
          self.fc = nn.Linear(feat_dim, 128)
          self.fc_actor1 = nn.Linear(128, self.action_dim)
          self.lin6 = nn.Softmax(dim=1) #注意是沿着那个维度计算
          self.fc2 = nn.Linear(feat_dim, 128)
          self.fc_critic1 = nn.Linear(128, 1)
          self.std = nn.Parameter(torch.zeros((1, self.action_dim)))

     def forward(self, img, depth, mpt, local_goal_point, angle):
          img_emb = self.feature_extractor_RGB(img)
          depth_emb = self.feature_extractor_depth(depth)
          mpt_emb = self.feature_extractor_mpt(mpt)
          obv_s = torch.cat((img_emb, depth_emb, mpt_emb), 1)
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
          
          output = self.fc(x)
          output = self.fc_actor1(output)
          mean = self.lin6(output)
          std = F.softplus(self.std).expand(mean.size(0), -1)

          return {
               'mean': mean,
               'std': std,
          }



class tfOptionNet(nn.Module):
     def __init__(self,
                    action_dim, droprate=0.5, feat_dim=2048):
          super(tfOptionNet, self).__init__()
          self.action_dim = action_dim
          self.droprate = droprate
          self.feature_extractor_RGB = ResNet(ResBlock).to(device0)
          self.feature_extractor_depth = ResNet(ResBlock).to(device0)
          self.feature_extractor_mpt = ResNet(ResBlock).to(device0)
          self.lin1 = nn.Linear(1, 256)
          self.lin2 = nn.Linear(1, 256)
          self.fc = nn.Linear(feat_dim, 128)
          self.fc_actor1 = nn.Linear(128, self.action_dim)
          self.lin6 = nn.Softmax(dim=1) #注意是沿着那个维度计算
          self.fc2 = nn.Linear(feat_dim, 128)
          self.fc_critic1 = nn.Linear(128, 1)
          self.std = nn.Parameter(torch.zeros((1, self.action_dim)))

     def forward(self, img, depth, mpt, local_goal_point, angle):
          img_emb = self.feature_extractor_RGB(img)
          depth_emb = self.feature_extractor_depth(depth)
          mpt_emb = self.feature_extractor_mpt(mpt)
          obv_s = torch.cat((img_emb, depth_emb, mpt_emb), 1)
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
          
          output = self.fc(x)
          output = self.fc_actor1(output)
          mean = self.lin6(output)
          std = F.softplus(self.std).expand(mean.size(0), -1)

          return {
               'mean': mean,
               'std': std,
          }




class testOptionNet1(nn.Module):
     def __init__(self,
                    action_dim, droprate=0.5, feat_dim=2048):
          super(testOptionNet1, self).__init__()
          self.action_dim = action_dim
          self.droprate = droprate
          self.feature_extractor_RGB = ResNet(ResBlock).to(device0)
          self.feature_extractor_depth = ResNet(ResBlock).to(device0)
          self.feature_extractor_mpt = ResNet(ResBlock).to(device0)
          self.lin1 = nn.Linear(1, 256)
          self.lin2 = nn.Linear(1, 256)
          self.fc = nn.Linear(feat_dim, 128)
          self.fc_actor1 = nn.Linear(128, self.action_dim)
          self.lin6 = nn.Softmax(dim=1) #注意是沿着那个维度计算
          self.fc2 = nn.Linear(feat_dim, 128)
          self.fc_critic1 = nn.Linear(128, 1)
          self.std = nn.Parameter(torch.zeros((1, self.action_dim)))

     def forward(self, img, depth, mpt, local_goal_point, angle):
          img_emb = self.feature_extractor_RGB(img)
          depth_emb = self.feature_extractor_depth(depth)
          mpt_emb = self.feature_extractor_mpt(mpt)
          obv_s = torch.cat((img_emb, depth_emb, mpt_emb), 1)
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
          
          output = self.fc(x)
          output = self.fc_actor1(output)
          mean = self.lin6(output)
          std = F.softplus(self.std).expand(mean.size(0), -1)

          return {
               'mean': mean,
               'std': std,
          }



class testOptionNet2(nn.Module):
     def __init__(self,
                    action_dim, droprate=0.5, feat_dim=2048):
          super(testOptionNet2, self).__init__()
          self.action_dim = action_dim
          self.droprate = droprate
          self.feature_extractor_RGB = ResNet(ResBlock).to(device0)
          self.feature_extractor_depth = ResNet(ResBlock).to(device0)
          self.feature_extractor_mpt = ResNet(ResBlock).to(device0)
          self.lin1 = nn.Linear(1, 256)
          self.lin2 = nn.Linear(1, 256)
          self.fc = nn.Linear(feat_dim, 128)
          self.fc_actor1 = nn.Linear(128, self.action_dim)
          self.lin6 = nn.Softmax(dim=1) #注意是沿着那个维度计算
          self.fc2 = nn.Linear(feat_dim, 128)
          self.fc_critic1 = nn.Linear(128, 1)
          self.std = nn.Parameter(torch.zeros((1, self.action_dim)))

     def forward(self, img, depth, mpt, local_goal_point, angle):
          img_emb = self.feature_extractor_RGB(img)
          depth_emb = self.feature_extractor_depth(depth)
          mpt_emb = self.feature_extractor_mpt(mpt)
          obv_s = torch.cat((img_emb, depth_emb, mpt_emb), 1)
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
          
          output = self.fc(x)
          output = self.fc_actor1(output)
          mean = self.lin6(output)
          std = F.softplus(self.std).expand(mean.size(0), -1)

          return {
               'mean': mean,
               'std': std,
          }



class testOptionNet3(nn.Module):
     def __init__(self,
                    action_dim, droprate=0.5, feat_dim=2048):
          super(testOptionNet3, self).__init__()
          self.action_dim = action_dim
          self.droprate = droprate
          self.feature_extractor_RGB = ResNet(ResBlock).to(device0)
          self.feature_extractor_depth = ResNet(ResBlock).to(device0)
          self.feature_extractor_mpt = ResNet(ResBlock).to(device0)
          self.lin1 = nn.Linear(1, 256)
          self.lin2 = nn.Linear(1, 256)
          self.fc = nn.Linear(feat_dim, 128)
          self.fc_actor1 = nn.Linear(128, self.action_dim)
          self.lin6 = nn.Softmax(dim=1) #注意是沿着那个维度计算
          self.fc2 = nn.Linear(feat_dim, 128)
          self.fc_critic1 = nn.Linear(128, 1)
          self.std = nn.Parameter(torch.zeros((1, self.action_dim)))

     def forward(self, img, depth, mpt, local_goal_point, angle):
          img_emb = self.feature_extractor_RGB(img)
          depth_emb = self.feature_extractor_depth(depth)
          mpt_emb = self.feature_extractor_mpt(mpt)
          obv_s = torch.cat((img_emb, depth_emb, mpt_emb), 1)
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
          
          output = self.fc(x)
          output = self.fc_actor1(output)
          mean = self.lin6(output)
          std = F.softplus(self.std).expand(mean.size(0), -1)

          return {
               'mean': mean,
               'std': std,
          }
     




class testOptionNet4(nn.Module):
     def __init__(self,
                    action_dim, droprate=0.5, feat_dim=2048):
          super(testOptionNet4, self).__init__()
          self.action_dim = action_dim
          self.droprate = droprate
          self.feature_extractor_RGB = ResNet(ResBlock).to(device0)
          self.feature_extractor_depth = ResNet(ResBlock).to(device0)
          self.feature_extractor_mpt = ResNet(ResBlock).to(device0)
          self.lin1 = nn.Linear(1, 256)
          self.lin2 = nn.Linear(1, 256)
          self.fc = nn.Linear(feat_dim, 128)
          self.fc_actor1 = nn.Linear(128, self.action_dim)
          self.lin6 = nn.Softmax(dim=1) #注意是沿着那个维度计算
          self.fc2 = nn.Linear(feat_dim, 128)
          self.fc_critic1 = nn.Linear(128, 1)
          self.std = nn.Parameter(torch.zeros((1, self.action_dim)))

     def forward(self, img, depth, mpt, local_goal_point, angle):
          img_emb = self.feature_extractor_RGB(img)
          depth_emb = self.feature_extractor_depth(depth)
          mpt_emb = self.feature_extractor_mpt(mpt)
          obv_s = torch.cat((img_emb, depth_emb, mpt_emb), 1)
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
          
          output = self.fc(x)
          output = self.fc_actor1(output)
          mean = self.lin6(output)
          std = F.softplus(self.std).expand(mean.size(0), -1)

          return {
               'mean': mean,
               'std': std,
          }




class testOptionNet5(nn.Module):
     def __init__(self,
                    action_dim, droprate=0.5, feat_dim=2048):
          super(testOptionNet5, self).__init__()
          self.action_dim = action_dim
          self.droprate = droprate
          self.feature_extractor_RGB = ResNet(ResBlock).to(device0)
          self.feature_extractor_depth = ResNet(ResBlock).to(device0)
          self.feature_extractor_mpt = ResNet(ResBlock).to(device0)
          self.lin1 = nn.Linear(1, 256)
          self.lin2 = nn.Linear(1, 256)
          self.fc = nn.Linear(feat_dim, 128)
          self.fc_actor1 = nn.Linear(128, self.action_dim)
          self.lin6 = nn.Softmax(dim=1) #注意是沿着那个维度计算
          self.fc2 = nn.Linear(feat_dim, 128)
          self.fc_critic1 = nn.Linear(128, 1)
          self.std = nn.Parameter(torch.zeros((1, self.action_dim)))

     def forward(self, img, depth, mpt, local_goal_point, angle):
          img_emb = self.feature_extractor_RGB(img)
          depth_emb = self.feature_extractor_depth(depth)
          mpt_emb = self.feature_extractor_mpt(mpt)
          obv_s = torch.cat((img_emb, depth_emb, mpt_emb), 1)
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
          
          output = self.fc(x)
          output = self.fc_actor1(output)
          mean = self.lin6(output)
          std = F.softplus(self.std).expand(mean.size(0), -1)

          return {
               'mean': mean,
               'std': std,
          }




class testOptionNet6(nn.Module):
     def __init__(self,
                    action_dim, droprate=0.5, feat_dim=2048):
          super(testOptionNet6, self).__init__()
          self.action_dim = action_dim
          self.droprate = droprate
          self.feature_extractor_RGB = ResNet(ResBlock).to(device0)
          self.feature_extractor_depth = ResNet(ResBlock).to(device0)
          self.feature_extractor_mpt = ResNet(ResBlock).to(device0)
          self.lin1 = nn.Linear(1, 256)
          self.lin2 = nn.Linear(1, 256)
          self.fc = nn.Linear(feat_dim, 128)
          self.fc_actor1 = nn.Linear(128, self.action_dim)
          self.lin6 = nn.Softmax(dim=1) #注意是沿着那个维度计算
          self.fc2 = nn.Linear(feat_dim, 128)
          self.fc_critic1 = nn.Linear(128, 1)
          self.std = nn.Parameter(torch.zeros((1, self.action_dim)))

     def forward(self, img, depth, mpt, local_goal_point, angle):
          img_emb = self.feature_extractor_RGB(img)
          depth_emb = self.feature_extractor_depth(depth)
          mpt_emb = self.feature_extractor_mpt(mpt)
          obv_s = torch.cat((img_emb, depth_emb, mpt_emb), 1)
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
          
          output = self.fc(x)
          output = self.fc_actor1(output)
          mean = self.lin6(output)
          std = F.softplus(self.std).expand(mean.size(0), -1)

          return {
               'mean': mean,
               'std': std,
          }




class Option_pi(nn.Module):
     def __init__(self,
                    num_options, droprate=0.5, feat_dim=2048):
          super(Option_pi, self).__init__()
          self.num_options = num_options
          self.droprate = droprate
          self.feature_extractor_RGB = ResNet(ResBlock).to(device0)
          self.feature_extractor_depth = ResNet(ResBlock).to(device0)
          self.feature_extractor_mpt = ResNet(ResBlock).to(device0)
          self.lin1 = nn.Linear(1, 256)
          self.lin2 = nn.Linear(1, 256)
          self.fc = nn.Linear(feat_dim, 128)
          self.fc_actor1 = nn.Linear(128, self.num_options)

     def forward(self, img, depth, mpt, local_goal_point, angle):
          img_emb = self.feature_extractor_RGB(img)
          depth_emb = self.feature_extractor_depth(depth)
          mpt_emb = self.feature_extractor_mpt(mpt)
          obv_s = torch.cat((img_emb, depth_emb, mpt_emb), 1)
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
          
          output = self.fc(x)
          output = self.fc_actor1(output)

          return output

class Option_q(nn.Module):
     def __init__(self,
                    num_options, droprate=0.5, feat_dim=2048):
          super(Option_q, self).__init__()
          self.num_options = num_options
          self.droprate = droprate
          self.feature_extractor_RGB = ResNet(ResBlock).to(device0)
          self.feature_extractor_depth = ResNet(ResBlock).to(device0)
          self.feature_extractor_mpt = ResNet(ResBlock).to(device0)
          self.lin1 = nn.Linear(1, 256)
          self.lin2 = nn.Linear(1, 256)
          self.fc = nn.Linear(feat_dim, 128)
          self.fc_actor1 = nn.Linear(128, self.num_options)

     def forward(self, img, depth, mpt, local_goal_point, angle):
          img_emb = self.feature_extractor_RGB(img)
          depth_emb = self.feature_extractor_depth(depth)
          mpt_emb = self.feature_extractor_mpt(mpt)
          obv_s = torch.cat((img_emb, depth_emb, mpt_emb), 1)
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
          
          output = self.fc(x)
          output = self.fc_actor1(output)

          return output


class OptionGaussianActorCriticNet(nn.Module):
     def __init__(self, action_dim, droprate=0.5, feat_dim=2048):
          super(OptionGaussianActorCriticNet, self).__init__()
          self.action_dim = action_dim
          self.option_safe = SafeOptionNet(action_dim = self.action_dim).to(device0)
          self.option_Collide = CollideOptionNet(action_dim = self.action_dim).to(device0)
          self.option_tf = tfOptionNet(action_dim = self.action_dim).to(device0)

          self.option_test1 = testOptionNet1(action_dim = self.action_dim).to(device0)
          self.option_test2 = testOptionNet2(action_dim = self.action_dim).to(device0)
          self.option_test3 = testOptionNet3(action_dim = self.action_dim).to(device0)
          self.option_test4 = testOptionNet4(action_dim = self.action_dim).to(device0)
          self.option_test5 = testOptionNet5(action_dim = self.action_dim).to(device0)
          self.option_test6 = testOptionNet6(action_dim = self.action_dim).to(device0)

          self.pi_o_net = Option_pi(num_options = 9).to(device0)
          self.q_o_net = Option_q(num_options = 9).to(device0)

     
     def forward(self, img, depth, mpt, local_goal_point, angle):
          mean = []
          std = []
          prediction = self.option_safe(img, depth, mpt, local_goal_point, angle)
          # print ('prediction[mean].shape: ', prediction['mean'].shape)  #torch.Size([1, 3])
          mean.append(prediction['mean'].unsqueeze(0))
          std.append(prediction['std'].unsqueeze(0))
          prediction = self.option_Collide(img, depth, mpt, local_goal_point, angle)
          mean.append(prediction['mean'].unsqueeze(0))
          std.append(prediction['std'].unsqueeze(0))
          prediction = self.option_tf(img, depth, mpt, local_goal_point, angle)
          mean.append(prediction['mean'].unsqueeze(0))
          std.append(prediction['std'].unsqueeze(0))
          prediction = self.option_test1(img, depth, mpt, local_goal_point, angle)
          mean.append(prediction['mean'].unsqueeze(0))
          std.append(prediction['std'].unsqueeze(0))
          prediction = self.option_test2(img, depth, mpt, local_goal_point, angle)
          mean.append(prediction['mean'].unsqueeze(0))
          std.append(prediction['std'].unsqueeze(0))
          prediction = self.option_test3(img, depth, mpt, local_goal_point, angle)
          mean.append(prediction['mean'].unsqueeze(0))
          std.append(prediction['std'].unsqueeze(0))
          prediction = self.option_test4(img, depth, mpt, local_goal_point, angle)
          mean.append(prediction['mean'].unsqueeze(0))
          std.append(prediction['std'].unsqueeze(0))
          prediction = self.option_test5(img, depth, mpt, local_goal_point, angle)
          mean.append(prediction['mean'].unsqueeze(0))
          std.append(prediction['std'].unsqueeze(0))
          prediction = self.option_test6(img, depth, mpt, local_goal_point, angle)
          mean.append(prediction['mean'].unsqueeze(0))
          std.append(prediction['std'].unsqueeze(0))
          mean = torch.cat(mean, dim=1)
          std = torch.cat(std, dim=1)

          phi_a = self.pi_o_net(img, depth, mpt, local_goal_point, angle)
          pi_o = F.softmax(phi_a, dim=-1)
          log_pi_o = F.log_softmax(phi_a, dim=-1)

          q_o = self.q_o_net(img, depth, mpt, local_goal_point, angle)


          return {'mean': mean,
                    'std': std,
                    'q_o': q_o,
                    'inter_pi': pi_o,
                    'log_inter_pi': log_pi_o}




# if __name__ == "__main__":
#      device0 = torch.device("cuda:0")
#      # img = torch.rand(1, 3, 256, 256)
#      model = Actor_Net(out_dim = 3).cuda()
#      # h1,h2,h3,h4,h5 = model_rgb(img.to(device0))       # RGBNet's output
#      summary(model, input_size=[(3, 320, 240), (3, 320, 240), (3, 320, 240), (1,2), (1,1)], batch_size = 1, device = 'cuda')


