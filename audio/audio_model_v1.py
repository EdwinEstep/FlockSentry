# tutorial for audio ML using librosa
# https://towardsdatascience.com/audio-deep-learning-made-simple-part-2-why-mel-spectrograms-perform-better-aad889a93505

# tutorial using torchaudio - tweaking as I go to accomodate different datasets
# https://colab.research.google.com/github/pytorch/tutorials/blob/gh-pages/_downloads/audio_classifier_tutorial.ipynb#scrollTo=hjZed3d1QBql

# some datasets included in audio file:
# https://github.com/zebular13/ChickenLanguageDataset
# https://www.robots.ox.ac.uk/~vgg/data/vggsound/
# https://research.google.com/audioset/balanced_train/chicken_rooster.html

# extra dataset - use APIv2 to interact, need user key:
# # https://freesound.org/docs/api/overview.html

# dataset YouTube ID's are preceeded by: youtu.be/

# tutorial start:
from email import header
import torch
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import Dataset
import torchaudio
import pandas as pd
import numpy as np
import youtube_dl
import os

from AudioSet import AudioSet
from AudioNet import AudioNet
from audio_train import train
from audio_test import test

# check if cuda gpu available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)

# import dataset
csvData = pd.read_csv('./audio/AudioSet_train.csv')


csv_path = './audio/AudioSet_train.csv'
file_path = './audio/AudioSet_Files'

# AudioSet class - formats data & gets filenames with matching labels
train_set = AudioSet(csv_path, file_path, "./audio/AudioSet_Files")
test_set = AudioSet(csv_path, file_path, [10])
print("Train set size: " + len(train_set))
print("Test set size: " + len(test_set))

kwargs = {'num_workers': 1, 'pin_memory': True} if device == 'cuda' else {} #needed for using datasets on gpu

train_loader = torch.utils.data.DataLoader(train_set, batch_size = 128, shuffle = True, **kwargs)
test_loader = torch.utils.data.DataLoader(test_set, batch_size = 128, shuffle = True, **kwargs)

# define the model
model = AudioNet()
model.to(device)
print(model)


# set training optimization and scheduler
optimizer = optim.Adam(model.parameters(), lr = 0.01, weight_decay = 0.0001)
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size = 20, gamma = 0.1)

# train and test model
log_interval = 20
for epoch in range(1, 41):
    if epoch == 31:
        print("First round of training complete. Setting learn rate to 0.001.")
    scheduler.step()
    train(model, epoch, optimizer, log_interval, train_loader, device)
    test(model, epoch, test_loader, device)
