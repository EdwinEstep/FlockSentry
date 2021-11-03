"""
Moves a random sample of one tenth of images and labels to the validation data
folder, and the rest to the training data folder.
"""

import shutil
from glob import glob
import os
import random

VALIDATION_PROPORTION = 0.1
YOLO_DATA_PATH = "../yolov5/data/"

label_fnames = glob("./*.txt")
n = int(len(label_fnames) * VALIDATION_PROPORTION)
validation_labels = random.sample(label_fnames, n)

try:
    shutil.rmtree(YOLO_DATA_PATH + "images/")
    shutil.rmtree(YOLO_DATA_PATH + "labels/")
except FileNotFoundError as e:  # We have already deleted the folders
    print(f"Not deleting nonexistent images/ and labels/ folders in {YOLO_DATA_PATH}.")
os.mkdir(YOLO_DATA_PATH + "images/")
os.mkdir(YOLO_DATA_PATH + "labels/")
os.mkdir(YOLO_DATA_PATH + "images/val")
os.mkdir(YOLO_DATA_PATH + "images/train")
os.mkdir(YOLO_DATA_PATH + "labels/val")
os.mkdir(YOLO_DATA_PATH + "labels/train")

for label_fname in label_fnames:
    img_fname = ""
    label_fname_root = label_fname.replace(".txt", "")
    if os.path.exists(f"{label_fname_root}.jpg"):
        img_fname = f"{label_fname_root}.jpg"
    elif os.path.exists(f"{label_fname_root}.jpeg"):
        img_fname = f"{label_fname_root}.jpeg"
    elif os.path.exists(f"{label_fname_root}.png"):
        img_fname = f"{label_fname_root}.png"
    else:
        raise RuntimeError(f"No matching .jpg or .png file for {label_fname}")

    if label_fname in validation_labels:
        print(img_fname, YOLO_DATA_PATH + "images/val/")
        shutil.copy2(img_fname, YOLO_DATA_PATH + "images/val/")
        shutil.copy2(label_fname, YOLO_DATA_PATH + "labels/val/")
    else:
        shutil.copy2(img_fname, YOLO_DATA_PATH + "images/train/")
        shutil.copy2(label_fname, YOLO_DATA_PATH + "labels/train/")
