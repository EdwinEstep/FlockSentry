## download_yt.py

The following packages are used for this tool:
- os
- subprocess
- pandas
- youtube-dl (installed using pip)

This file downloads the videos (and audio) contained in the `AudioSet_train.csv` file, and places them in the `/audio/AudioSet_Files` folder.

This should also work with the `AudioSet_eval.csv` file. With minor changes, this should work with any dataset we have. In particular, we would need to change:
- Save directory
- Amount of values pulled from .csv file
- Labels of those values
- Parameters in the `downloadCommand` variable