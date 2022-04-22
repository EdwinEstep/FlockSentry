import os
import cv2
import numpy as np
import math
import random


def sigmoid(x):
    return 1 / (1 + math.exp(-x))


NAMES = ["chicken", "fox", "bird", "person", "cat", "dog"]


def draw_boxes(img_fname):
    img = cv2.imread(img_fname)
    labels_fname = img_fname.replace("images/", "labels/").split(".", 1)[0] + ".txt"
    with open(labels_fname, "r") as f:
        img_height, img_width, _ = img.shape
        line_width = int(0.01 * img_height)
        for line in f.readlines():
            linesplit = line.split(" ")
            object_class = int(linesplit[0])
            label = NAMES[object_class]
            box_width = int(float(linesplit[3]) * img_width)
            box_height = int(float(linesplit[4]) * img_height)
            x1 = int(float(linesplit[1]) * img_width) - box_width // 2
            y1 = int(float(linesplit[2]) * img_height) - box_height // 2
            x2 = x1 + box_width
            y2 = y1 + box_height
            r = sigmoid(hash(label) >> 63)
            g = sigmoid(hash(label[1:] + label[0]) >> 63)
            b = sigmoid(hash(label[2:] + label[0:2]) >> 63)
            brightness = math.sqrt(r * r + g * g + b * b)
            r = int(r * 255 / brightness / 2 + 128)
            g = int(g * 255 / brightness / 2 + 128)
            b = int(b * 255 / brightness / 2 + 128)
            cv2.putText(
                img,
                label,
                (min(x1, x2) + 5 * line_width, min(y1, y2) + 5 * line_width),
                cv2.FONT_HERSHEY_SIMPLEX,
                line_width / 5,
                (b, g, r),
                max(line_width // 2, 1),
                cv2.LINE_AA,
            )
            cv2.rectangle(img, (x1, y1), (x2, y2), (b, g, r), line_width)

    window_name = f"Annotated {img_fname}"
    cv2.namedWindow(window_name, cv2.WINDOW_KEEPRATIO)
    cv2.resizeWindow(window_name, 800, 400)
    cv2.imshow(window_name, img)


if __name__ == "__main__":
    # train_path = "training/yolov5/data/images/train/"
    train_path = "training/22apr22-chickens/"
    # val_path = "training/yolov5/data/images/val/"
    val_path = ""
    filenames = os.listdir(train_path)
    if os.path.exists(val_path):
        filenames.extend(os.listdir(val_path))
    filenames = [
        f for f in filenames if (f.endswith(".png") or f.endswith(".jpg") or f.endswith(".jpeg"))
    ]
    random.shuffle(filenames)
    for img_fname in filenames:
        if os.path.exists(train_path + img_fname):
            print(f"Showing {train_path + img_fname}")
            draw_boxes(train_path + img_fname)
        elif os.path.exists(val_path + img_fname):
            print(f"Showing {val_path + img_fname}")
            draw_boxes(val_path + img_fname)
        else:
            continue

        if cv2.waitKey(0) == ord("q"):
            break
        cv2.destroyAllWindows()
