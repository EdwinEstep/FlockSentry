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
import math 


def sigmoid(x):
    return 1 / (1 + math.exp(-x))


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
        CAMERA = cv2.VideoCapture(0)
        if not CAMERA.isOpened():
            raise RuntimeError("Could not open camera")

    if MODEL is None:
        MODEL = torch.hub.load('ultralytics/yolov5', 'custom', path="model_train/best.pt", force_reload=True)

    while True:
        _, img = CAMERA.read()
        
        results = MODEL(img)
        # print(results.pandas().xyxy[0])
        df = results.pandas().xyxy[0]
        for i in df.index:
            xmin, ymin = int(df["xmin"][i]), int(df["ymin"][i])
            xmax, ymax = int(df["xmax"][i]), int(df["ymax"][i])
            label = df["name"][i]
            r = sigmoid(hash(label) / (1 << 63))
            g = sigmoid(hash(label[1:] + label[0]) / (1 << 63))
            b = sigmoid(hash(label[2:] + label[0:2]) / (1 << 63))
            brightness = math.sqrt(r * r + g * g + b * b)
            r = int(r*255/brightness)
            g = int(g*255/brightness)
            b = int(b*255/brightness)
            cv2.putText(img, label, (xmin, ymin), cv2.FONT_HERSHEY_SIMPLEX, 1, (b, g, r), 1, cv2.LINE_AA)
            cv2.rectangle(img, (xmin, ymin), (xmax, ymax), (b, g, r), 2)

        _, buf = cv2.imencode(".jpg", img ) #encode as jpg
        frame = buf.tobytes()
        yield (b"--frame\r\n" b"Content-Type: image/jpeg\r\n\r\n" + frame + b"\r\n")


@app.route("/video_feed")
def video_feed():
    """Video streaming route. Put this in the src attribute of an img tag."""
    return Response(gen(), mimetype="multipart/x-mixed-replace; boundary=frame")


@app.route("/status")
def status():
    return "FLOCK SPOOKED" if random.random() >= 0.5 else "all is well with the flock"


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8080, debug=True, threaded=True)
