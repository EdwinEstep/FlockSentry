import torch.nn.functional as F
import torch.nn as nn

# neural net class to process raw audio
# supposed to be like M5 network architecture, https://arxiv.org/pdf/1610.00087.pdf

# make sure to update to AudioSet
#   from tutorial:
'''  
    Our model's first filter is length 80 so when processing audio sampled at 8kHz 
    the receptive field is around 10ms. This size is similar to speech processing 
    applications that often use receptive fields ranging from 20ms to 40ms.
'''

# TODO: preprocess the data; make it all be standard 8kHz, and the same length
# split up positive ID's between shortened files
# take samples every ~5 seconds
class AudioNet(nn.Module):
    def __init__(self):
        super(AudioNet, self).__init__()
        self.conv1 = nn.Conv1d(1, 80, 3, 4) # batch size 3, receptive field 80, kernel size 4,  stride 1
        self.bn1 = nn.BatchNorm1d(3)
        self.pool1 = nn.MaxPool1d(4)
        self.conv2 = nn.Conv1d(3, 3, 3)
        self.bn2 = nn.BatchNorm1d(128)
        self.pool2 = nn.MaxPool1d(4)
        self.conv3 = nn.Conv1d(3, 6, 3)
        self.bn3 = nn.BatchNorm1d(6)
        self.pool3 = nn.MaxPool1d(4)
        self.conv4 = nn.Conv1d(6, 12, 3)
        self.bn4 = nn.BatchNorm1d(12)
        self.pool4 = nn.MaxPool1d(4)
        self.avgPool = nn.AvgPool1d(30) #input should be 512x30 so this outputs a 512x1
        self.fc1 = nn.Linear(12, 10)
        
    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(self.bn1(x))
        x = self.pool1(x)
        x = self.conv2(x)
        x = F.relu(self.bn2(x))
        x = self.pool2(x)
        x = self.conv3(x)
        x = F.relu(self.bn3(x))
        x = self.pool3(x)
        x = self.conv4(x)
        x = F.relu(self.bn4(x))
        x = self.pool4(x)
        x = self.avgPool(x)
        x = x.permute(0, 2, 1) #change the 512x1 to 1x512
        x = self.fc1(x)
        return F.log_softmax(x, dim = 2)

