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
        
        # returns data, sample_rate
        # data (sample_file[0]) has format [channels, audio frames]; sample_rate is sample_file[1]
        sample_file, sample_rate = torchaudio.load(audio_folder + "/" + self.file_names[0])
        
        # sample file should have 1 channel, 496312 frames
        print("Sample File 0: ", sample_file[0])
        print("Length of Sample File 0: ", len(sample_file[0]))
        print("Sample Rate: ", sample_rate) 

        ###################################################################################
        # TODO: check and make sure this is a valid change from DownmixMono()
        #self.mixer = torch.mean(sample_file[0])
        '''
        Notes:
        look at tensor.unfold
        keepdim = True ?
        '''
        self.mixer = torch.mean(sample_file[0], dim=0)#.unsqueeze(0)
        print("**** Self.Mixer **** \n", self.mixer)


        self.audio_folder = audio_folder

        # maybe try to resize here?
        



    def __getitem__(self, index):
        #format the file path and load the file
        # TODO: change when more Audio Files available (select folder from audio files folder)
        #path = self.file_path + "fold" + str(self.folders[index]) + "/" + self.file_names[index]
        
        path = self.file_path + "/" + self.file_names[index]
        
        try:
            sound = torchaudio.load(path) #, normalization = True) # removed out = None
        except:
            print("Could not open file at: ", path)
            

        # downsample to 8kHz
        soundData = self.mixer
        # TODO:
        '''
        Look into torchaudio.transforms.resample -- resamples the frequency
        Could be simpler easy fix?
        Diff btwn .transforms and .functional resampling functions
        '''
        '''
        # pad the data with 0's
        tempData = torch.zeros([1, 160000]) #tempData accounts for audio clips that are too short
        if soundData.numel() < 160000:
            tempData[:soundData.numel()] = soundData[:]
        else:
            tempData[:] = soundData[:160000]
        
        soundData = tempData
        '''

        effects = [
            ['gain', '-n'],  # normalises to 0dB
            #['pitch', '5'],  # 5 cent pitch shift
            ['rate', '8000'],  # resample to 8000 Hz
            #['pad', '0', '1.5'],  # add 1.5 seconds silence at the end
        ]

        # change sampling rate to 8 kHz
        waveform, sample_rate = torchaudio.sox_effects.apply_effects_tensor(sound[0], sound[1], effects, channels_first=True)
        print("Shape of waveform: ", waveform.shape)

        # downmix to one channel
        self.mixer = torch.mean(waveform, dim=0).unsqueeze(0)
        print("Shape of self.mixer: ", self.mixer.shape)

        # pad to the same length (do max length minus the size of current data)
        #tempData = torch.nn.functional.pad(self.mixer, (0, self.mixer.shape[1]))
        #print(tempData)
        #print("Shape of temp data: ", tempData.shape)

        # downsample to 8kHz
        #resampled_sound = torchaudio.transforms.Resample(orig_freq=sound[1], new_freq=8000)

        # get the newly formatted sound
        #soundFormatted = resampled_sound(self.mixer) # (sound[0])
        soundFormatted = self.mixer
        print("SoundFormatted shape: ", soundFormatted.shape)
        return soundFormatted, self.labels[index]
    
    def __len__(self):
        return len(self.file_names)
