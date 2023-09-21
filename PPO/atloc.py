import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init
from PPO.att import AttentionBlock

class FourDirectionalLSTM(nn.Module):
    def __init__(self, seq_size, origin_feat_size, hidden_size):
        super(FourDirectionalLSTM, self).__init__()
        self.feat_size = origin_feat_size // seq_size
        self.seq_size = seq_size
        self.hidden_size = hidden_size
        self.lstm_rightleft = nn.LSTM(self.feat_size, self.hidden_size, batch_first=True, bidirectional=True)
        self.lstm_downup = nn.LSTM(self.seq_size, self.hidden_size, batch_first=True, bidirectional=True)

    def init_hidden_(self, batch_size, device):
        return (torch.randn(2, batch_size, self.hidden_size).to(device),
                torch.randn(2, batch_size, self.hidden_size).to(device))

    def forward(self, x):
        batch_size = x.size(0)
        x_rightleft = x.view(batch_size, self.seq_size, self.feat_size)
        x_downup = x_rightleft.transpose(1, 2)
        hidden_rightleft = self.init_hidden_(batch_size, x.device)
        hidden_downup = self.init_hidden_(batch_size, x.device)
        _, (hidden_state_lr, _) = self.lstm_rightleft(x_rightleft, hidden_rightleft)
        _, (hidden_state_ud, _) = self.lstm_downup(x_downup, hidden_downup)
        hlr_fw = hidden_state_lr[0, :, :]
        hlr_bw = hidden_state_lr[1, :, :]
        hud_fw = hidden_state_ud[0, :, :]
        hud_bw = hidden_state_ud[1, :, :]
        return torch.cat([hlr_fw, hlr_bw, hud_fw, hud_bw], dim=1)

class AtLoc(nn.Module):
    def __init__(self, feature_extractor, out_dim, droprate=0.5, feat_dim=2048, lstm=False):
        super(AtLoc, self).__init__()
        self.droprate = droprate
        self.lstm = lstm
        self.out_dim = out_dim

        # replace the last FC layer in feature extractor
        self.feature_extractor = feature_extractor

        if self.lstm:
            self.lstm4dir = FourDirectionalLSTM(seq_size=32, origin_feat_size=feat_dim, hidden_size=256)
            self.fc_xyz = nn.Linear(feat_dim // 2, self.out_dim)
          #   self.fc_wpqr = nn.Linear(feat_dim // 2, 3)
        else:
            self.att = AttentionBlock(feat_dim)
            self.fc_xyz = nn.Linear(feat_dim, self.out_dim)
          #   self.fc_wpqr = nn.Linear(feat_dim, 3)



    def forward(self, x):
        x = self.feature_extractor(x)
        x = F.relu(x)
        print ('x_1.shape: ', x.shape)
        y = x.view(x.size(0), -1)
        print ('y.shape: ', y.shape)

        if self.lstm:
            x = self.lstm4dir(x)
        else:
            x = self.att(x.view(x.size(0), -1))

        if self.droprate > 0:
            x = F.dropout(x, p=self.droprate)

        xyz = self.fc_xyz(x)
     #    wpqr = self.fc_wpqr(x)
     #    return torch.cat((xyz, wpqr), 1)
        return xyz

class AtLocPlus(nn.Module):
    def __init__(self, atlocplus):
        super(AtLocPlus, self).__init__()
        self.atlocplus = atlocplus

    def forward(self, x):
        s = x.size()
        x = x.view(-1, *s[2:])
        poses = self.atlocplus(x)
        poses = poses.view(s[0], s[1], -1)
        return poses



