import os
import cv2
import numpy as np
import math
import random


def sigmoid(x):
    return 1 / (1 + math.exp(-x))


NAMES = [  # class names
    # Labels in our dataset
    "chicken",
    "fox",
    # "housepet",  # TODO: remove housepet in favor of cat and dog
    # labels in COCO dataset
    "bird",
    "person",
    # "cat",
    "dog",
    # "horse",
    # "sheep",
    # "cow",
    # "bear",
    # More labels in our dataset. Confusing, I know.
    # "bobcat",
    # "mountain_lion",
    # "raccoon",
    # "coyote",
]


def draw_boxes(img_fname):
    img = cv2.imread(img_fname)
    labels_fname = img_fname.replace("images/", "labels/").split(".", 1)[0] + ".txt"
    with open(labels_fname, "r") as f:
        img_height, img_width, _ = img.shape
        line_width = int(0.005 * img_height)
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
            r = int(r * 255 / brightness)
            g = int(g * 255 / brightness)
            b = int(b * 255 / brightness)
            cv2.putText(
                img,
                label,
                (min(x1, x2), min(y1, y2)),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (b, g, r),
                1,
                cv2.LINE_AA,
            )
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), line_width)

    window_name = f"Annotated {img_fname}"
    cv2.namedWindow(window_name, cv2.WINDOW_KEEPRATIO)
    cv2.resizeWindow(window_name, 800, 400)
    cv2.imshow(window_name, img)


if __name__ == "__main__":
    train_path = "training/new_data/"
    filenames = os.listdir(train_path)
    filenames = [f for f in filenames if f.endswith(".txt")]
    for fname in filenames:
        if os.path.exists(train_path + fname):
            print(f"Editing {train_path + fname}")
            with open(train_path + fname, "r") as f:
                lines = f.readlines()
                lines = [[float(n) for n in line.split(" ")] for line in lines]
                for line in lines:
                    line[0] = int(line[0])
                    line[3] = abs(line[3])
                    line[4] = abs(line[4])
            with open(train_path + fname, "w") as f:
                f.writelines([" ".join([str(n) for n in line]) + "\n" for line in lines])
        else:
            print(f"not found {train_path + fname}")
            continue

        if cv2.waitKey(0) == ord("q"):
            break
        cv2.destroyAllWindows()
