import torch
import torchaudio
import pandas as pd
from torch.utils.data import Dataset
import os

# AudioSet formattor class
# change to match AudioSet dataset - try to make samples same length or same sampling freq
# from tutorial...
'''
# convert files to torchaudio tensors
#   ensure files are mono-channel
# choose audio model input size (32000?)
# downsample data based on audio Hz and length in milliseconds to fit input size
#   pad extra space missed by sampling with 0's

# torchaudio.load() a .wav file
# returns tuple with the tensor and sampling frequency of file
'''

class AudioSet(Dataset):

#wrapper for AudioSet dataset
    # Argument List
    #  path to the AudioSet csv file
    #  path to the AudioSet audio files
    #  For future use: list of folders to use in the dataset

    
    def __init__(self, csv_path, file_path, audio_folder):
        csvData = pd.read_csv(csv_path)
        #initialize lists to hold file names, labels, and folders
        self.file_names = []    # youtube ID's
        self.labels = []        # positive labels
        # self.folders = [] # add later when more Audio Files are available

        # loop through CSV (skipping header)
        for i in range(1,len(csvData)):
            # if the youtube ID is in a filename in the audiofiles folder...
            # if csvData.iloc[i, 0] in audio_folder:
            for filename in os.listdir(audio_folder):
                if csvData.iloc[i, 0] in filename:
                    self.file_names.append(filename)  # filename ending with youtube ID
                    self.labels.append(csvData.iloc[i, 4::])    # labels
                    # self.folders.append(csvData.iloc[0, :])   # add later when more Audio Files are available
                
        self.file_path = file_path
        # self.mixer = torchaudio.transforms.DownmixMono() # Convert audio files to one channel # DownmixMono deprecated
        self.mixer = torch.mean(self.file_names[0])
        self.audio_folder = audio_folder

        print("Found filenames: \n" + self.file_names + "\nFound Labels: \n" + self.labels)
        
    def __getitem__(self, index):
        #format the file path and load the file
        # TODO: change when more Audio Files available (select folder from audio files folder)
        #path = self.file_path + "fold" + str(self.folders[index]) + "/" + self.file_names[index]
        
        path = self.file_path + "/" + self.file_names[index]
        sound = torchaudio.load(path, out = None, normalization = True)
        
        # load returns a tensor with the sound data and the sampling frequency
        soundData = self.mixer(sound[0])

        # downsample the audio to ~8kHz - might have to change based on average audio sampling
        tempData = torch.zeros([160000, 1]) #tempData accounts for audio clips that are too short
        if soundData.numel() < 160000:
            tempData[:soundData.numel()] = soundData[:]
        else:
            tempData[:] = soundData[:160000]
        
        soundData = tempData
        soundFormatted = torch.zeros([32000, 1])
        soundFormatted[:32000] = soundData[::5] # take every fifth sample of soundData
        soundFormatted = soundFormatted.permute(1, 0)
        return soundFormatted, self.labels[index]
    
    def __len__(self):
        return len(self.file_names)
