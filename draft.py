import numpy as np
import numpy.matlib as npm
import torch
from network.network_TD3 import *

import pandas as pd

dataframe = pd.DataFrame({'a': [25, 12, 15, 14],
                   'b': [5, 7, 13, 12]})
dataframe["c"] = [1, 1, 1, 1]
print(dataframe)
dataframe["d"] = []
print(dataframe)

# actors = nn.ModuleList(Actor(2).to(device0) for actor in range(3)).to(device0)

# temp = (range(3), actors)
# for o,a in zip(range(3), range(3)):
#     print(o)
    # print(a)
# option_index = torch.tensor([[1, 2, 3], [4, 5, 6], [7, 9, 8]])
# batch_s = torch.randn(3, 20, 20, 20)
# max_index = torch.argmax(option_index, dim=1)


# print(max_index)
# for o in range(3):
#     indx_o = (max_index == o)
#     print(indx_o)
#     print(indx_o.shape)
#     batch_s_o = batch_s[indx_o]
#     print(batch_s_o.shape)
    
# print(p.shape[0])

# p = p.numpy()
# # print(p.reshape(1, 
# # p = np.array([[1, 2, 3], [4, 5, 6]])
# row, col = p.shape


# p_sum = np.reshape(np.sum(p, axis=1), (row, 1))
# print('p_sum.shape: ', p_sum.shape)
# p_normalized = p  / npm.repmat(p_sum, 1, col)
# print('p_normalized.shape: ', p_normalized.shape)
# p_cumsum = np.matrix(np.cumsum( p_normalized, axis=1))
# # print(p_cumsum[0])
# rand = npm.repmat(np.random.random((row, 1)), 1, col)
# # print(rand[0])
# o_softmax = np.argmax(p_cumsum >= rand, axis=1)
# print(o_softmax.shape)