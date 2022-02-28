import os

# from here: https://stackoverflow.com/questions/26741116/python-extract-wav-from-video-file

VIDEOS_PATH = '/Users/emilyburns/Documents/school/Spring2022/Flock Sentry/FlockSentry/audio/AudioSet_Files'
VIDEOS_EXTENSION = '.mp4'   # extension of the video
AUDIO_EXT = 'wav'           # audio output type

EXTRACT_VIDEO_COMMAND = ('ffmpeg -i "{from_video_path}" '
                         '-f {audio_ext} '
                         '-vn "{to_audio_path}"')

os.chdir(VIDEOS_PATH)
files = os.listdir(VIDEOS_PATH)
for f in files:
    if not f.endswith(VIDEOS_EXTENSION):
        continue

    audio_file_name = '{}.{}'.format(f, AUDIO_EXT)
    command = EXTRACT_VIDEO_COMMAND.format(
        from_video_path=f, audio_ext=AUDIO_EXT, to_audio_path=audio_file_name,
    )
    os.system(command)