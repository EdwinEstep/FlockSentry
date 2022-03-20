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
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import Dataset
import torchaudio
import pandas as pd
import numpy as np
import youtube_dl
import os

# check if cuda gpu available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)

# import dataset
csvData = pd.read_csv('./audio/AudioSet_train.csv')
#print(csvData.iloc[0, :])
#print(csvData)

ytLinks = pd.read_csv('./audio/AudioSet_train.csv', usecols=[0], header = None, index_col=None)
ytLinks = ytLinks.iloc[1:,0]
print(ytLinks)

os.chdir('./audio/AudioSet_Files')
for url in ytLinks:
    #print(url)
    yt = "https://youtu.be/" + url

    try:
        ydl_opts = {}
        with youtube_dl.YoutubeDL(ydl_opts) as ydl:
            ydl.download([yt])
    except:
        print("Video Unavailable")

     

print("out")

