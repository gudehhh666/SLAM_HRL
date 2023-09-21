import torch
import torch.nn as nn
import numpy as np

# # rnn = nn.LSTM(10, 20, 2)
# # input = torch.randn(5, 3, 10)
# # h0 = torch.randn(2, 3, 20)
# # c0 = torch.randn(2, 3, 20)
# # output, (hn, cn) = rnn(input, (h0, c0))

# # print (output.shape)

# rnn = nn.LSTM(10, 20, 2)
# input = torch.randn(3, 10)
# print (input.shape)
# input2 = torch.unsqueeze(input, 0)
# print (input2.shape)
# # h0 = torch.randn(2, 3, 20)
# # c0 = torch.randn(2, 3, 20)
# # output, (hn, cn) = rnn(input, (h0, c0))
# output, (hn, cn) = rnn(input2)

# print (output.shape)

# device0 = torch.device("cuda:0")
# img = np.random.rand(5, 3, 320, 240)
# img = torch.from_numpy(img).float().to(device0)

# obv_s = torch.cat((img, img, img), 1)
# print (obv_s.shape)


# xx = []
# xx.append(1)
# xx.append(0)
# xx.append(3)
# xx.append(1)
# print (xx)
# xx.pop(1)
# print (xx)




# import random

# random_action = np.random.rand(1, 2)
# number1 = random.random()
# random_action[0,0] = (-0.5) * number1
# number2 = 2 * random.random() - 1
# random_action[0,1] = 0.3 * number2
# print ('random_action: ', random_action)










# device0 = torch.device("cuda:0")

# def _to_tensor(v):
#      if torch.is_tensor(v):
#           return v
#      elif isinstance(v, np.ndarray):
#           return torch.from_numpy(v)
#      else:
#           return torch.tensor(v, dtype=torch.float)
#           # return torch.as_tensor(v, dtype=torch.float)

# img = np.random.rand(1, 3, 320, 240)
# img = torch.from_numpy(img).float()


# image_tensor = [3 for x in range(0, 5)]

# image_tensor[0] = _to_tensor(img.squeeze(0)).float()
# image_tensor[1] = _to_tensor(img.squeeze(0)).float()
# image_tensor[2] = _to_tensor(img.squeeze(0)).float()
# image_tensor[3] = _to_tensor(img.squeeze(0)).float()
# image_tensor[4] = _to_tensor(img.squeeze(0)).float()

# index = np.random.choice(5, size=3)
# print ('index: ', index)

# # batch_image_tensor = torch.tensor(image_tensor[index], dtype=torch.float)
# # print ('batch_image_tensor.shape: ', batch_image_tensor.shape)

# batch_image_tensor = []
# for i in range(len(index)):
#      print ('index[i]: ', index[i])
#      batch_image_tensor.append(image_tensor[index[i]])
# batch_image_tensor_t = torch.stack(batch_image_tensor, dim=0).to(device=device0)

# print ('batch_image_tensor_t.shape: ', batch_image_tensor_t.shape)  #torch.Size([3, 3, 320, 240])





print ('np.log(2)： ', np.log(2))   #0.6931471805599453

print ('np.log(0.033)： ', np.log(0.033))  #-3.4112477175156566

print ('np.log(0.01)： ', np.log(0.01))   #-4.605170185988091
print ('np.exp(np.log(0.01))： ', np.exp(np.log(0.01)))  

target_entropy = np.log(0.033) * (0.6)
print ('target_entropy: ', target_entropy)  #-2.046748630509394


device0 = torch.device("cuda:0")
depth = torch.rand(64, 1).to(device0)
print (torch.mean(depth))




# probs = np.random.rand(1, 3)
# probs = torch.from_numpy(probs).float()
# print ('probs: ', probs)  #tensor([[0.0105, 0.0404, 0.9492]], device='cuda:0', grad_fn=<SoftmaxBackward>)
# print ('probs.shape: ', probs.shape)   #torch.Size([1, 3])
# action_dist = torch.distributions.Categorical(probs)
# action = action_dist.sample()
# print ('action: ', action)    #tensor([1], device='cuda:0')
# print ('action.shape: ', action.shape)  #torch.Size([1])
# action_ = action.item()
# print ('action_: ', action_)


# import torch
# import torch.nn as nn
# import numpy as np
# import torch.nn.functional as F
# from PPO.model_resnet18_fuse import ResNet, ResBlock, ResNet_RGB
# # from model_resnet18_fuse import ResNet, ResBlock, ResNet_RGB
# import torch.nn.init
# # from PPO.att import AttentionBlock
# # from torchsummary import summary
# from torch.distributions import Normal

# device0 = torch.device("cuda:0")


# class PolicyNetContinuous(torch.nn.Module):
#      def __init__(self, state_dim, hidden_dim, action_dim, action_bound):
#           super(PolicyNetContinuous, self).__init__()
#           self.fc1 = torch.nn.Linear(state_dim, hidden_dim)
#           self.fc_mu = torch.nn.Linear(hidden_dim, action_dim)
#           self.fc_std = torch.nn.Linear(hidden_dim, action_dim)
#           self.action_bound = action_bound

#      def forward(self, x):
#           x = F.relu(self.fc1(x))
#           mu = self.fc_mu(x)
#           std = F.softplus(self.fc_std(x))
#           dist = Normal(mu, std)
#           normal_sample = dist.rsample()  # rsample()是重参数化采样
#           log_prob = dist.log_prob(normal_sample)
#           action = torch.tanh(normal_sample)
#           # 计算tanh_normal分布的对数概率密度
#           log_prob = log_prob - torch.log(1 - torch.tanh(action).pow(2) + 1e-7)
#           action = action * self.action_bound
#           return action, log_prob

# input_test = np.random.rand(5, 240)
# input_test = torch.from_numpy(input_test).float()


# model = PolicyNetContinuous(240, 120, 3, )



action_dim = 6
max_action = 1.0
noise_std = 0.1 * max_action
s = np.random.normal(0, noise_std, size=action_dim)

print ('s: ', s)



b = np.random.normal(0, 0.1 * 0.5)
print ('b: ', b)


# action_ = np.random.rand(2)
# print ('action_: ', action_)


import random
action_ = np.random.rand(2)
# action_step = np.random.rand(1, 2)
# number1 = random.random()
number1 = 2 * random.random() - 1
action_[0] = 0.5 * number1
# action_step[0,0] = (-0.5) * number1
number2 = 2 * random.random() - 1
action_[1] = 0.3 * number2
print ('action_: ', action_)
