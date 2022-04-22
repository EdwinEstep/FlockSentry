import cv2
import os

data_path = "./training/all_image_data/"
fnames = sorted([f for f in os.listdir(data_path) if "_" in f and f.endswith(".txt")])
for fname in fnames:
    lines_to_write = []
    with open(data_path + fname, "r") as f:
        lines_to_write = [
            l
            for l in f.readlines()
            if 0.0 < float(l.split(" ")[1]) < 1.0
            and 0.0 < float(l.split(" ")[2]) < 1.0
            and 0.0 < float(l.split(" ")[3]) < 1.0
            and 0.0 < float(l.split(" ")[4]) < 1.0
        ]

    if not lines_to_write:
        img = cv2.imread(data_path + fname.replace(".txt", ".png"))
        cv2.imshow(fname, img)
        print(f"Delete {data_path + fname}? [y/N]")
        response = cv2.waitKey(0)
        if response == "y":
            os.remove(data_path + fname)
            os.remove(data_path + fname.replace(".txt", ".png"))

    with open(data_path + fname, "w") as f:
        f.writelines(lines_to_write)
