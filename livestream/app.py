"""
Made following [this guide](https://www.instructables.com/Video-Streaming-Web-Server/) and
[this guide](https://towardsdatascience.com/video-streaming-in-web-browsers-with-opencv-flask-93a38846fe00)
"""
from flask import Flask, render_template, Response
import time
import cv2

app = Flask(__name__)


CAMERA = None
LAST_REFRESH = None


@app.route("/")
def index():
    """Video streaming home page."""
    return render_template("index.html")


def gen():
    global CAMERA, LAST_REFRESH

    """Video streaming generator function."""
    if CAMERA is None:
        CAMERA = cv2.VideoCapture(0)
        if not CAMERA.isOpened():
            raise RuntimeError("Could not open camera")
    while True:
        if LAST_REFRESH is not None:
            print(f"refresh time: {time.time() - LAST_REFRESH}")
        # frame = camera.get_frame()
        _, img = CAMERA.read()
        _, buf = cv2.imencode(".jpg", img)
        frame = buf.tobytes()
        LAST_REFRESH = time.time()
        yield (b"--frame\r\n" b"Content-Type: image/jpeg\r\n\r\n" + frame + b"\r\n")


@app.route("/video_feed")
def video_feed():
    """Video streaming route. Put this in the src attribute of an img tag."""
    return Response(gen(), mimetype="multipart/x-mixed-replace; boundary=frame")


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8080, debug=True, threaded=True)
