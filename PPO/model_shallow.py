import torch
import torch.nn as nn
import numpy as np



class Shallow(nn.Module):
     def __init__(self,n_class=2):
          super(Shallow, self).__init__()

          # original image's size = 320*240*3

          # conv1
          self.conv1_1 = nn.Conv2d(3, 32, kernel_size=5, stride=2)
          self.conv1_2 = nn.Conv2d(32, 32, kernel_size=3, stride=1)

          # conv2
          self.conv2_1 = nn.Conv2d(32, 64, kernel_size=5, stride=2)
          self.conv2_2 = nn.Conv2d(64, 64, kernel_size=3, stride=1)





if __name__ == "__main__":
     device0 = torch.device("cuda:0")
     model = ResNet(ResBlock).to(device0)
     img = np.random.rand(3, 3, 256, 256)
     img = torch.from_numpy(img).float().to(device0)
     depth = np.random.rand(3, 1, 256, 256)
     depth = torch.from_numpy(depth).float().to(device0)
     mpt = np.random.rand(3, 1, 256, 256)
     mpt = torch.from_numpy(mpt).float().to(device0)
     obv_s = torch.cat((img, depth, mpt), 1)
     print ('obv_s.shape: ', obv_s.shape)
     output1 = model(obv_s)
     print (output1.shape)




