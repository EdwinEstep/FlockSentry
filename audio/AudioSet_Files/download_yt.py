import pandas as pd
import youtube_dl
import subprocess
import os

# import youtube video id's, clip start times, and clip end times from dataset
csvData = pd.read_csv('./audio/AudioSet_train.csv', usecols=[0,1,2], header = None, index_col=None)

# trim header
ytLinks = csvData.iloc[1:,0]
start_times = csvData.iloc[1:,1]
end_times = csvData.iloc[1:,2]
# print(ytLinks)
# print(start_times)
# print(end_times)

# change to directory to download files in
os.chdir('./audio/AudioSet_Files')

# iterate through youtube links
for n in range(len(ytLinks)):
    # get values
    yt = "http://youtu.be/" + ytLinks[n+1]
    start_time = start_times[n+1]
    end_time = end_times[n+1]

    # get direct youtube link - not needed for the youtu.be links
    """ bashCommand = "youtube-dl --get-url " + yt 
    process = subprocess.Popen(bashCommand.split(), stdout=subprocess.PIPE)
    output, error = process.communicate()
    output = output.decode("utf-8")
    #print(output)
    print("Successfully got video link") """

    # download links to current dir
    downloadCommand = "youtube-dl -f bestvideo+bestaudio " + yt + ' --external-downloader ffmpeg --external-downloader-args "-ss ' + start_time + ' -t ' + end_time + '"'
    # print(downloadCommand)
    os.system(downloadCommand)
    print("///// Process Comlete /////")



print("Done")