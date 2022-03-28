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

app = Flask(__name__)


CAMERA = None
CHICKENS_SPOOKED = False
LAST_UPDATE = None


@app.route("/")
def index():
    """Video streaming home page."""
    return render_template("index.html", spooked=CHICKENS_SPOOKED)


def gen():
    global CAMERA, CHICKENS_SPOOKED, LAST_UPDATE

    model = torch.hub.load('ultralytics/yolov5', 'custom', path="model_train/best.pt", force_reload=True)

    """Video streaming generator function."""
    if CAMERA is None:
        CAMERA = cv2.VideoCapture(0)
        if not CAMERA.isOpened():
            raise RuntimeError("Could not open camera")
    while True:
        # frame = camera.get_frame()
        _, img = CAMERA.read()
        _, buf = cv2.imencode(".jpg", img ) #encode as jpg
        
        results = model(img)
        print(results.pandas().xyxy[0])

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
