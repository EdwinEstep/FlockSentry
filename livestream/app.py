"""
Made following [this guide](https://www.instructables.com/Video-Streaming-Web-Server/) and
[this guide](https://towardsdatascience.com/video-streaming-in-web-browsers-with-opencv-flask-93a38846fe00)
"""
from flask import Flask, render_template, Response
from flask_cors import CORS
import random
import time
import cv2
import torch
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


def iou(box1, box2):
    left1, top1, width1, height1 = box1
    left2, top2, width2, height2 = box2
    x_overlap = 0
    if left1 < left2:
        x_overlap = max(0, left1 + width1 - left2 - max(0, left1 + width1 - left2 - width2))
    else:
        x_overlap = max(0, left2 + width2 - left1 - max(0, left2 + width2 - left1 - width1))
    y_overlap = 0
    if top1 < top2:
        y_overlap = max(0, top1 + height1 - top2 - max(0, top1 + height1 - top2 - height2))
    else:
        y_overlap = max(0, top2 + height2 - top1 - max(0, top2 + height2 - top1 - height1))
    intersect = x_overlap * y_overlap
    union = width1 * height1 + width2 * height2 - intersect
    return intersect / union


class CustomTracker(object):
    """A wrapper of an OpenCV KCF object tracker that keeps a record of its age and confidence"""

    def __init__(self, img, label, confidence, bbox):
        self.tracker = cv2.legacy.TrackerKCF_create()
        self.tracker.init(img, bbox)
        self.label = label
        self.confidence = confidence
        self.age = 0
        self.path = [self._center_of_bbox(bbox)]
        self.latest_bbox = bbox
        self.vx = 0
        self.vy = 0
        self.t = time.time()

    def _center_of_bbox(self, bbox):
        l, t, w, h = bbox
        return round((2 * l + w) / 2), round((2 * t + h) / 2)

    def reset(self, img, confidence, bbox):
        """Provide a new bounding box and confidence to the tracker"""
        self.tracker.init(img, bbox)
        self.confidence = confidence
        self.age = 0
        self.latest_bbox = bbox
        self.path = [self._center_of_bbox(bbox)]

    def update(self, img):
        """Calculate a bounding box from the tracker"""
        self.age += 1
        _, bbox = self.tracker.update(img)
        if bbox[0] > 2 and bbox[1] > 2:
            self.path.append(self._center_of_bbox(bbox))
        self.latest_bbox = bbox
        dt = time.time() - self.t
        self.t = time.time()
        if len(self.path) > 1:
            self.vx = (self.path[-1][0] - self.path[-2][0]) / dt
            self.vy = (self.path[-1][1] - self.path[-2][1]) / dt
        else:
            self.vx = 0
            self.vy = 0
        return _, bbox


app = Flask(__name__)
CORS(app)


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
        CAMERA = cv2.VideoCapture(gstreamer_pipeline(), cv2.CAP_GSTREAMER)

    out = cv2.VideoWriter("./output_1.avi", -1, 30.0, (640, 480))

    chicken_trackers = []
    agitation_short_avg = 0
    agitation_long_avg = 100
    flock_count = 0

    while CAMERA.isOpened():
        ok, img = CAMERA.read()
        if not ok:
            break

        # img = cv2.resize(img, (640, 480), interpolation=cv2.INTER_LINEAR)

        # img = torch.from_numpy(img).to(device)

        results = MODEL(img)
        df = results.pandas().xyxy[0]
        chickens_detected = []
        foxes_detected = []
        for i in df.index:
            if df["class"][i] == 0:
                # if df["confidence"][i] > 0.:
                chickens_detected.append(
                    (
                        df["name"][i],
                        df["confidence"][i],
                        df["xmin"][i],
                        df["ymin"][i],
                        df["xmax"][i] - df["xmin"][i],
                        df["ymax"][i] - df["ymin"][i],
                    )
                )
            elif df["class"][i] == 1:
                # if df["confidence"][i] > 0.:
                foxes_detected.append(
                    (
                        df["name"][i],
                        df["confidence"][i],
                        df["xmin"][i],
                        df["ymin"][i],
                        df["xmax"][i] - df["xmin"][i],
                        df["ymax"][i] - df["ymin"][i],
                    )
                )

        # Detect agitation
        velocities = [math.sqrt(t.vx * t.vx + t.vy * t.vy) for t in chicken_trackers]
        agitation = sum(velocities) / max(1, len(velocities))
        # concerning_velocities = [v for v in velocities if v > 80]
        # agitation = len(concerning_velocities) / max(1, len(velocities))
        flock_count = 0.99 * flock_count + 0.01 * len(chicken_trackers)
        agitation_short_avg = 0.8 * agitation_short_avg + 0.2 * agitation
        agitation_long_avg = 0.999 * agitation_long_avg + 0.001 * agitation
        # print(
        #     f"long: {agitation_long_avg:.2f}, short: {agitation_short_avg:.2f}, flock count: {flock_count:.2f}"
        # )
        CHICKENS_SPOOKED = (
            # fox in scene
            len(foxes_detected) > 0
            # higher than normal agitation
            or (agitation_long_avg > 10 and agitation_short_avg > 10 * agitation_long_avg)
            # flock disappears
            or (flock_count > 2 and len(chickens_detected) < 0.5 * flock_count)
        )

        # Remove chicken and fox trackers that are too old
        chicken_trackers_to_delete = []
        chickens_detected_to_delete = []
        chickens_remembered = []
        for i, tracker in enumerate(chicken_trackers):
            _, box = tracker.update(img)
            # print(
            #     "remembered chicken no.",
            #     i,
            #     "with confidence",
            #     tracker.confidence,
            #     "and age",
            #     tracker.age,
            # )
            if tracker.age > 120 * tracker.confidence * tracker.confidence:
                # print("deleting no.", i, "from trackers")
                chicken_trackers_to_delete.append(i)
                continue
            for j, box2 in enumerate(chickens_detected):
                if iou(box, box2[2:]) > 0.4:
                    # print("deleting no.", j, "from detections")
                    tracker.reset(img, box2[1], box2[2:])
                    if j not in chickens_detected_to_delete:
                        chickens_detected_to_delete.append(j)
            chickens_remembered.append((box[0], conf, *box))

        chicken_trackers_to_delete = sorted(chicken_trackers_to_delete, reverse=True)
        for i in chicken_trackers_to_delete:
            del chicken_trackers[i]
        chickens_detected_to_delete = sorted(chickens_detected_to_delete, reverse=True)
        for i in chickens_detected_to_delete:
            del chickens_detected[i]

        # Sort by confidence.  Least confident will be left out.
        chickens_detected = sorted(chickens_detected, key=lambda c: c[1], reverse=True)
        for label, conf, l, t, w, h in chickens_detected:
            if len(chicken_trackers) < 10:  # Limit the number of trackers to a reasonable amount
                # print("initializing chicken no.", len(chicken_trackers), "with confidence", conf)
                tracker = CustomTracker(img, label, conf, (l, t, w, h))
                chicken_trackers.append(tracker)

        for tracker in chicken_trackers:
            l, t, w, h = tracker.latest_bbox
            if w == 0 or h == 0:
                continue
            name = tracker.label
            draw_box(img, l, t, l + w, t + h, name)
            last_point = None
            for point in tracker.path:
                if last_point is None:
                    last_point = point
                    continue
                cv2.line(img, point, last_point, (0, 255, 0), thickness=2, lineType=cv2.LINE_AA)
                last_point = point

        # for name, _, l, t, w, h in chickens_detected + chickens_remembered:
        #     draw_box(img, l, t, l + w, t + h, name)

        out.write(img)
        _, buf = cv2.imencode(".jpg", img)
        frame = buf.tobytes()
        yield (b"--frame\r\n" b"Content-Type: image/jpeg\r\n\r\n" + frame + b"\r\n")
    out.release()
    CAMERA.release()
    CAMERA = None
    print("camera closed")


@app.route("/video_feed")
def video_feed():
    """Video streaming route. Put this in the src attribute of an img tag."""
    return Response(gen(), mimetype="multipart/x-mixed-replace; boundary=frame")


@app.route("/status")
def status():
    global CHICKENS_SPOOKED
    return "FLOCK SPOOKED" if CHICKENS_SPOOKED else "all is well with the flock"


if __name__ == "__main__":
    if MODEL is None:
        # MODEL = torch.hub.load(
        #     "/home/flocksentry/Desktop/yolov5", "custom", path="model_train/best.pt", source="local"
        # )
        MODEL = torch.hub.load("ultralytics/yolov5", "custom", path="model_train/best.pt")
        device = torch.device("cuda:0")
        MODEL.to(device)
    app.run(host="0.0.0.0", port=8080, debug=True, threaded=True)
