import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
from PPO.model_resnet18_fuse import ResNet, ResBlock, ResNet_RGB
import torch.nn.init
from PPO.att import AttentionBlock
# from torchsummary import summary

device0 = torch.device("cuda:0")


class Actor_Net(nn.Module):
     def __init__(self, out_dim, droprate=0.5, feat_dim=2048):
          super(Actor_Net, self).__init__()
          self.out_dim = out_dim
          self.droprate = droprate
          self.out_dim = out_dim
          # self.feature_extractor_RGB = ResNet_RGB(ResBlock).to(device0)
          self.feature_extractor_RGB = ResNet(ResBlock).to(device0)
          self.feature_extractor_depth = ResNet(ResBlock).to(device0)
          self.feature_extractor_mpt = ResNet(ResBlock).to(device0)
          self.lin1 = nn.Linear(1, 256)
          self.lin2 = nn.Linear(1, 256)
          self.att = AttentionBlock(feat_dim)
          self.fc1 = nn.Linear(feat_dim, 256)
          #lstm layer
          self.lstm = nn.LSTM(256, 256)
          self.hidden_cell = None

          self.fc2 = nn.Linear(256, self.out_dim)
          self.lin6 = nn.Softmax(dim=2) #注意是沿着那个维度计算
     
     # def get_init_state(self, batch_size, device):
     #      self.hidden_cell = (torch.zeros(1, batch_size, 256).to(device), torch.zeros(1, batch_size, 256).to(device))

     
     def forward(self, img, depth, mpt, local_goal_point, angle, h_in):
          # batch_size = img.shape[0]
          # device = img.device
          # if (self.hidden_cell is None) or (batch_size != self.hidden_cell[0].shape[1]):
          #      self.get_init_state(batch_size, device)

          img_emb = self.feature_extractor_RGB(img)
          depth_emb = self.feature_extractor_depth(depth)
          mpt_emb = self.feature_extractor_mpt(mpt)
          obv_s = torch.cat((img_emb, depth_emb, mpt_emb), 1)
          local_goal_point = local_goal_point.float()
          local_goal_point = local_goal_point.view(-1, 1).to(device0)
          local_goal_point_emb1 = F.leaky_relu(self.lin1(local_goal_point))
          # print ('angle.shape: ', angle)
          angle = angle.view(-1, 1)
          # print ('angle.shape: ', angle)
          angle_emb1 = F.leaky_relu(self.lin2(angle))
          scalar_values = torch.cat((local_goal_point_emb1, angle_emb1), 1)
          x = torch.cat((obv_s, scalar_values), 1)

          # print ('x.shape: ', x.shape)  #torch.Size([1, 2048])
          # # print ('x.size(0): ', x.size(0))  #1
          # # print ('x.size(1): ', x.size(1))  #2048
          # # print ('x.size(2): ', x.size(2))  #IndexError: Dimension out of range (expected to be in range of [-2, 1], but got 2)
          # y = x.view(x.size(0), -1)
          # print ('y.shape: ', y.shape)  #torch.Size([1, 2048])
          # y2 = x.view(1, x.size(0), -1)  
          # print ('y2.shape: ', y2.shape)  #y2.shape:  torch.Size([1, 1, 2048])

          x = self.att(x.view(x.size(0), -1))
          if self.droprate > 0:
               x = F.dropout(x, p=self.droprate)
          # print ('x.shape: ', x.shape)
          embed1 = self.fc1(x)
          # print ('embed1.shape: ', embed1.shape)  #torch.Size([1, 256])

          # #lstm
          # embed2 = embed1.view(1, embed1.size(0), -1)
          # # print ('embed2.shape: ', embed2.shape)
          # lstm_out, self.hidden_cell = self.lstm(embed2, self.hidden_cell)
          # # print ('lstm_out.shape: ', lstm_out.shape)
          # lstm_out = torch.squeeze(lstm_out, 0)

          # #lstm
          # embed2 = embed1.view(1, embed1.size(0), -1)
          # # print ('embed2.shape: ', embed2.shape)
          # lstm_out, self.hidden_cell = self.lstm(embed2, self.hidden_cell)
          # # print ('lstm_out.shape: ', lstm_out.shape)
          # lstm_out = torch.squeeze(lstm_out, 0)

          #lstm
          embed2 = embed1.view(-1, 1, 256)
          lstm_out, lstm_hidden = self.lstm(embed2, h_in)
          output = self.fc2(lstm_out)
          # print ('output.shape: ', output.shape)
          out = self.lin6(output)
          # print ('out: ', out)
          # print ('out.shape: ', out.shape)


          # output = self.fc2(embed1)
          # out = self.lin6(output)
          # print ('Actor out.shape: ', out.shape)
          # # print ('out[0].shape: ', out[0].shape)
          # print ('out: ', out)
          # # out2 = torch.squeeze(out, 0)
          # # print ('out2: ', out2)
          # # print ('out2.shape: ', out2.shape)
          return out, lstm_hidden


class Critic_Net(nn.Module):
     def __init__(self, out_dim, droprate=0.0, feat_dim=2048):
          super(Critic_Net, self).__init__()
          self.out_dim = out_dim
          self.droprate = droprate
          self.out_dim = out_dim
          # self.feature_extractor_RGB = ResNet_RGB(ResBlock).to(device0)
          self.feature_extractor_RGB = ResNet(ResBlock).to(device0)
          self.feature_extractor_depth = ResNet(ResBlock).to(device0)
          self.feature_extractor_mpt = ResNet(ResBlock).to(device0)
          self.lin1 = nn.Linear(1, 256)
          self.lin2 = nn.Linear(1, 256)
          self.att = AttentionBlock(feat_dim)
          self.fc1 = nn.Linear(feat_dim, 256)
          #lstm layer
          self.lstm = nn.LSTM(256, 256, 1)
          self.hidden_cell = None

          self.fc2 = nn.Linear(256, self.out_dim)
     
     # def get_init_state(self, batch_size, device):
     #      self.hidden_cell = (torch.zeros(1, batch_size, 256).to(device), torch.zeros(1, batch_size, 256).to(device))
     
     def forward(self, img, depth, mpt, local_goal_point, angle, h_in):
          # batch_size = img.shape[0]
          # device = img.device
          # if (self.hidden_cell is None) or (batch_size != self.hidden_cell[0].shape[1]):
          #      self.get_init_state(batch_size, device)

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
          x = self.att(x.view(x.size(0), -1))
          if self.droprate > 0:
               x = F.dropout(x, p=self.droprate)
          embed1 = self.fc1(x)
          # embed2 = embed1.view(1, embed1.size(0), -1)
          # lstm_out, self.hidden_cell = self.lstm(embed2, self.hidden_cell)
          # lstm_out = torch.squeeze(lstm_out, 0)
          # output = self.fc2(lstm_out)
          embed2 = embed1.view(-1, 1, 256)
          lstm_out, lstm_hidden = self.lstm(embed2, h_in)
          output = self.fc2(lstm_out)
          # print ('Critic output.shape: ', output.shape)
          # print ('output[0].shape: ', output[0].shape)
          # print ('output: ', output)
          # output2 = torch.squeeze(output, 0)
          return output


# if __name__ == "__main__":
#      device0 = torch.device("cuda:0")
#      # img = torch.rand(1, 3, 256, 256)
#      model = Actor_Net(out_dim = 3).cuda()
#      # h1,h2,h3,h4,h5 = model_rgb(img.to(device0))       # RGBNet's output
#      summary(model, input_size=[(3, 320, 240), (3, 320, 240), (3, 320, 240), (1,2), (1,1)], batch_size = 1, device = 'cuda')


