import os
import cv2
import numpy as np


def draw_boxes(img_fname):
    img = cv2.imread(img_fname)
    labels_fname = img_fname.replace("images/", "labels/").split(".", 1)[0] + ".txt"
    with open(labels_fname, "r") as f:
        img_height, img_width, _ = img.shape
        line_width = int(0.005 * img_height)
        for line in f.readlines():
            linesplit = line.split(" ")
            object_class = int(linesplit[0])
            box_width = int(float(linesplit[3]) * img_width)
            box_height = int(float(linesplit[4]) * img_height)
            x1 = int(float(linesplit[1]) * img_width) - box_width // 2
            y1 = int(float(linesplit[2]) * img_height) - box_height // 2
            x2 = x1 + box_width
            y2 = y1 + box_height
            img = cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), line_width)

    window_name = f"Annotated {img_fname}"
    cv2.namedWindow(window_name, cv2.WINDOW_KEEPRATIO)
    cv2.resizeWindow(window_name, 800, 400)
    cv2.imshow(window_name, img)


if __name__ == "__main__":
    train_path = "yolov5/data/images/train/"
    val_path = "yolov5/data/images/val/"
    for img_fname in os.listdir(train_path):
        draw_boxes(train_path + img_fname)
    for img_fname in os.listdir(val_path):
        draw_boxes(val_path + img_fname)

    cv2.waitKey(0)
    cv2.destroyAllWindows()
