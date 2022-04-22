import os
import cv2
import math
import copy

data_path = "./training/new_data/"


def sigmoid(x):
    return 1 / (1 + math.exp(-x))


X_SCALE = 1
Y_SCALE = 1
X_OFFSET = 0
Y_OFFSET = 0
WINDOW_TITLE = "corrupted image"
NAMES = [  # class names
    # Labels in our dataset
    "chicken",
    "fox",
    "housepet",  # TODO: remove housepet in favor of cat and dog
    # labels in COCO dataset
    "bird",
    "person",
    "cat",
    "dog",
    "horse",
    "sheep",
    "cow",
    "bear",
    # More labels in our dataset. Confusing, I know.
    "bobcat",
    "mountain_lion",
    "raccoon",
    "coyote",
]


def on_x_scale_trackbar(val):
    global X_SCALE
    X_SCALE = val


def on_y_scale_trackbar(val):
    global Y_SCALE
    Y_SCALE = val


def on_x_offset_trackbar(val):
    global X_OFFSET
    X_OFFSET = val - 500


def on_y_offset_trackbar(val):
    global Y_OFFSET
    Y_OFFSET = val - 500


if __name__ == "__main__":
    cv2.namedWindow(WINDOW_TITLE)
    cv2.createTrackbar("x scale", WINDOW_TITLE, 500, 10000, on_x_scale_trackbar)
    cv2.createTrackbar("y scale", WINDOW_TITLE, 500, 10000, on_y_scale_trackbar)
    cv2.createTrackbar("x offset", WINDOW_TITLE, 500, 1000, on_x_offset_trackbar)
    cv2.createTrackbar("y offset", WINDOW_TITLE, 500, 1000, on_y_offset_trackbar)

    fnames = sorted([f for f in os.listdir(data_path) if "_" in f and f.endswith(".txt")])
    i = 0
    last_code = None
    while i < len(fnames):
        plain_img = cv2.imread(data_path + fnames[i].replace(".txt", ".png"))
        img_height, img_width, _ = plain_img.shape
        print(f"tweaking {data_path + fnames[i]} with width {img_width} and height {img_height}")

        while True:
            x_scale = max(X_SCALE / 500, 1 / 500)
            y_scale = max(Y_SCALE / 500, 1 / 500)
            x_offset = X_OFFSET
            y_offset = Y_OFFSET
            img = copy.copy(plain_img)
            with open(data_path + fnames[i], "r") as f:
                lines_to_write = []
                for line in f.readlines():
                    category, l, t, r, b = line.split()
                    xcenter = (float(l) + float(r)) / 2 * x_scale + x_offset / 500
                    ycenter = (float(t) + float(b)) / 2 * y_scale + y_offset / 500
                    w = abs(float(r) - float(l)) * x_scale
                    h = abs(float(b) - float(t)) * y_scale
                    lines_to_write.append((category, xcenter, ycenter, w, h))
            for line in lines_to_write:
                object_class = int(line[0])
                label = NAMES[object_class]
                box_width = int(float(line[3]) * img_width)
                box_height = int(float(line[4]) * img_height)
                x1 = int(float(line[1]) * img_width) - box_width // 2
                y1 = int(float(line[2]) * img_height) - box_height // 2
                x2 = x1 + box_width
                y2 = y1 + box_height
                r = sigmoid(hash(label) / (1 << 63))
                g = sigmoid(hash(label[1:] + label[0]) / (1 << 63))
                b = sigmoid(hash(label[2:] + label[0:2]) / (1 << 63))
                brightness = math.sqrt(r * r + g * g + b * b)
                r = int(r * 255 / brightness)
                g = int(g * 255 / brightness)
                b = int(b * 255 / brightness)
                cv2.putText(
                    img,
                    f"{len(lines_to_write)} boxes",
                    (10, 20),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (0, 255, 0),
                    1,
                    cv2.LINE_AA,
                )
                cv2.putText(
                    img,
                    label,
                    (min(x1, x2), min(y1, y2)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (b, g, r),
                    1,
                    cv2.LINE_AA,
                )
                cv2.rectangle(img, (x1, y1), (x2, y2), (b, g, r), 3)
                cv2.rectangle(  # Show where the rect would be if unmodified
                    img,
                    (
                        int(float(line[1]) - float(line[3]) / 2),
                        int(float(line[2]) - float(line[4]) / 2),
                    ),
                    (
                        int(float(line[1]) + float(line[3]) / 2),
                        int(float(line[2]) + float(line[4]) / 2),
                    ),
                    (255, 0, 255),
                    3,
                )
                cv2.imshow(WINDOW_TITLE, img)

            code = cv2.waitKey(50)
            print(last_code)
            if code == ord("q") and last_code != ord("q"):
                exit()
            if code == ord("y") and last_code != ord("y"):
                last_code = code
                with open(data_path + fnames[i], "w") as f:
                    for line in lines_to_write:
                        f.write(f"{line[0]} {line[1]} {line[2]} {line[3]} {line[4]}\n")
                break
            if code == ord("n") and last_code != ord("n"):  # next
                last_code = code
                break
            if code == ord("p") and last_code != ord("p"):  # previous
                last_code = code
                i = max(i - 2, -1)
                break
            last_code = code
        i += 1
