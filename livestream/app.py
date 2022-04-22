"""
Made following [this guide](https://www.instructables.com/Video-Streaming-Web-Server/) and
[this guide](https://towardsdatascience.com/video-streaming-in-web-browsers-with-opencv-flask-93a38846fe00)
"""
from flask import Flask, render_template, Response
import random
import time
import io
import cv2
from PIL import Image
import torch
import torchvision
import math


def gstreamer_pipeline(
    sensor_id=0,
    capture_width=1920,
    capture_height=1080,
    display_width=960,
    display_height=540,
    framerate=30,
    flip_method=0,
):
    return (
        "nvarguscamerasrc sensor-id=%d !"
        "video/x-raw(memory:NVMM), width=(int)%d, height=(int)%d, framerate=(fraction)%d/1 ! "
        "nvvidconv flip-method=%d ! "
        "video/x-raw, width=(int)%d, height=(int)%d, format=(string)BGRx ! "
        "videoconvert ! "
        "video/x-raw, format=(string)BGR ! appsink"
        % (
            sensor_id,
            capture_width,
            capture_height,
            framerate,
            flip_method,
            display_width,
            display_height,
        )
    )


def sigmoid(x):
    return 1 / (1 + math.exp(-x))


def draw_box(img, x1, y1, x2, y2, label):
    xmin, ymin = int(min(x1, x2)), int(min(y1, y2))
    xmax, ymax = int(max(x1, x2)), int(max(y1, y2))
    r = sigmoid(hash(label) / (1 << 63))
    g = sigmoid(hash(label[1:] + label[0]) / (1 << 63))
    b = sigmoid(hash(label[2:] + label[0:2]) / (1 << 63))
    brightness = math.sqrt(r * r + g * g + b * b)
    r = int(r * 255 / brightness)
    g = int(g * 255 / brightness)
    b = int(b * 255 / brightness)
    cv2.putText(img, label, (xmin, ymin), cv2.FONT_HERSHEY_SIMPLEX, 1, (b, g, r), 1, cv2.LINE_AA)
    cv2.rectangle(img, (xmin, ymin), (xmax, ymax), (b, g, r), 2)


app = Flask(__name__)


CAMERA = None
CHICKENS_SPOOKED = False
LAST_UPDATE = None
MODEL = None


@app.route("/")
def index():
    """Video streaming home page."""
    return render_template("index.html", spooked=CHICKENS_SPOOKED)


def gen():
    global CAMERA, CHICKENS_SPOOKED, LAST_UPDATE, MODEL

    """Video streaming generator function."""
    if CAMERA is None:
        # CAMERA = cv2.VideoCapture(gstreamer_pipeline(), cv2.CAP_GSTREAMER)
        CAMERA = cv2.VideoCapture(0)

    while CAMERA.isOpened():
        ok, img = CAMERA.read()
        if not ok:
            break

        # h, w, _ = img.shape
        # new_h = 640 // max(w, h) * h
        # new_w = 640 // max(w, h) * w
        # cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
        # img = torch.from_numpy(img).half() / 255
        # img = [img]

        # img = torch.from_numpy(img).to(device)

        results = MODEL(img)
        df = results.pandas().xyxy[0]
        for i in df.index:
            if df["class"][i] in [0, 1]:
                draw_box(
                    img, df["xmin"][i], df["xmax"][i], df["ymin"][i], df["ymax"][i], df["name"][i]
                )
        _, buf = cv2.imencode(".jpg", img)  # encode as jpg
        frame = buf.tobytes()
        yield (b"--frame\r\n" b"Content-Type: image/jpeg\r\n\r\n" + frame + b"\r\n")
    CAMERA.release()
    CAMERA = None
    print("camera closed")


@app.route("/video_feed")
def video_feed():
    """Video streaming route. Put this in the src attribute of an img tag."""
    return Response(gen(), mimetype="multipart/x-mixed-replace; boundary=frame")


@app.route("/status")
def status():
    return "FLOCK SPOOKED" if random.random() >= 0.5 else "all is well with the flock"


if __name__ == "__main__":
    if MODEL is None:
        # MODEL = torch.hub.load(
        #     "/home/flocksentry/Desktop/yolov5", "custom", path="model_train/best.pt", source="local"
        # )
        MODEL = torch.hub.load("ultralytics/yolov5", "custom", path="model_train/best.pt")
        device = torch.device("cuda:0")
        MODEL.to(device)
    app.run(host="0.0.0.0", port=8080, debug=True, threaded=True)
