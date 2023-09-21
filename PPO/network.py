
import torch.nn.functional as F
import torch.nn as nn
import torch
device0 = torch.device("cuda:0")


class Net(nn.Module):
    def __init__(self, out_dim):
        super(Net, self).__init__()

        # self.a_dim = a_dim
        self.out_dim = out_dim

        self.conv1 = nn.Conv2d(in_channels=5, out_channels=20, kernel_size=11, stride=1, padding=0)
        # self.conv1_drop = nn.Dropout2d()
        self.bnc1 = torch.nn.BatchNorm2d(20, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.conv2 = nn.Conv2d(in_channels=20, out_channels=36, kernel_size=8, stride=1, padding=0)
        # self.conv2_drop = nn.Dropout2d()
        self.bnc2 = torch.nn.BatchNorm2d(36, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.conv3 = nn.Conv2d(in_channels=36, out_channels=50, kernel_size=5, stride=1, padding=0)
        self.conv3_drop = nn.Dropout2d()
        # self.lin1 = nn.Linear(206250, 1000)
        self.lin1 = nn.Linear(43750, 1000)
        self.lin2 = nn.Linear(1729, 50)
        
        self.lin3 = nn.Linear(53, self.out_dim)
        self.lin4 = nn.Softmax(dim=0) #注意是沿着那个维度计算

    def forward(self, rgbdp, local_map, local_goal_point, angle):
        x1 = F.relu(F.max_pool2d(self.bnc1(self.conv1(rgbdp)), 2))
        x2 = F.relu(F.max_pool2d(self.bnc2(self.conv2(x1)), 2))
        x3 = F.relu(F.max_pool2d(self.conv3_drop(self.conv3(x2)), 2))
        # x4 = x3.view(-1, 206250)
        x4 = x3.view(-1, 43750)
        x5 = F.relu(self.lin1(x4))
        local_map = local_map.view(-1, 729)
        # print('local_map_shape', local_map.shape)
        # print('x5', x5.shape)
        x5 = torch.cat((x5, local_map), 1)
        # print('x5', x5.shape)
        x6 = F.relu(self.lin2(x5))
        # print(x6.shape)
        local_goal_point = local_goal_point.view(-1, 2)
        angle = torch.tensor(angle).to(device0)
        local_goal_point = torch.tensor(local_goal_point).to(device0)
        # print (local_goal_point.shape)
        x7 = torch.cat((x6, local_goal_point), 1)
        # print ('x7.shape: ', x7.shape)
        # print ('angle.shape: ', angle.shape)
        # print ('angle: ', angle)
        angle = angle.view(-1, 1)
        # print ('angle.shape: ', angle.shape)
        x8 = torch.cat((x7, angle), 1)
        # print ('x8.shape: ', x8.shape)
        # print ('x8.type: ', type(x8))
        x8 = x8.to(torch.float32)
        # print ('x8.shape: ', x8.shape)

        x9 = F.relu(self.lin3(x8))

        out = self.lin4(x9)
        # out = F.softmax(x8, 3)
        # out = F.relu(self.lin3(x8))


        return out

