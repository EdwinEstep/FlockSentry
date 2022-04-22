import cv2
import sys
from random import randint

tracker_types = ['BOOSTING', 'MIL', 'KCF', 'TLD', 'MEDIANFLOW', 'MOSSE', 'CSRT']

def create_tracker_by_name(tracker_type):
    if tracker_type == tracker_types[0]:
        tracker = cv2.TrackerBoosting_create()
    elif tracker_type == tracker_types[1]:
        tracker = cv2.TrackerMIL_create()
    elif tracker_type == tracker_types[2]:
        tracker = cv2.TrackerKCF_create()
    elif tracker_type == tracker_types[3]:
        tracker = cv2.TrackerTLD_create()
    elif tracker_type == tracker_types[4]:
        tracker = cv2.TrackerMedianFlow_create()
    elif tracker_type == tracker_types[5]:
        tracker = cv2.TrackerMOSSE_create()
    elif tracker_type == tracker_types[6]:
        tracker = cv2.TrackerCSRT_create()
    else:
        tracker = None
        print('Invalid name. Try: ')
        for t in tracker_types:
            print(t)
    return tracker

# print(create_tracker_by_name('CSRT'))

video = cv2.VideoCapture('Videos/running.mp4')
if not video.isOpened():
    print("Error loading video!")
    sys.exit()
ok, frame = video.read()

bboxes = []
colors = []

# this code is used to manually set bounding boxes in video
# our code should already handle this
while True:
    bbox = cv2.selectROI('MultiTracker', frame)
    bboxes.append(bbox)
    colors.append((randint(0,255), randint(0,255), randint(0,255)))
    print('Press Q to quit and start tracking')
    print('press any other key to select the next object')
    k = cv2.waitKey(0) & 0xFF
    if k == 113: #Q - quit
        break

print(bboxes)
print(colors)
old_boxes = []

tracker_type = 'CSRT'
multi_tracker = cv2.MultiTracker_create()
for bbox in bboxes:
    # frame is the first frame of the video
    multi_tracker.add(create_tracker_by_name(tracker_type), frame, bbox)

while video.isOpened():
    ok, frame = video.read()
    if not ok:
        break

    ok, boxes = multi_tracker.update(frame) #update positions
    # draw tracked objects
    if (old_boxes == []):
        ok, old_boxes = multi_tracker.update(frame)
    for i, newbox in enumerate(boxes):
        # p1 = (int(newbox[0]), int(newbox[1]), int(newbox[2]), int(newbox[3]))
        # print("new below")
        # print(p1)
        # find dif between positions
        for j, oldbox in enumerate(old_boxes):
            # uncomment these for sanity check
            # print("old below")
            # print(int(oldbox[0]), int(oldbox[1]), int(oldbox[2]), int(oldbox[3]))
            # print("difference")
            diff = abs(int(newbox[0]-oldbox[0])), abs(int(newbox[1]-oldbox[1])), abs(int(newbox[2]-oldbox[2])), abs(int(newbox[3]-oldbox[3]))
            # print(diff)
            if (diff[0] > 4 or diff[1] > 4 or diff[2] > 4 or diff[3] > 4):
                # the 4 above should be changed to whatever threshold we find concerning
                print("DANGER IS EMINENT")

    # update box position
    for i, new_box in enumerate(boxes):
        (x, y, w, h) = [int(v) for v in new_box]
        cv2.rectangle(frame, (x, y), (x + w, y + h), colors[i], 2)

    old_boxes = boxes

    cv2.imshow('MultiTracker', frame)
    if cv2.waitKey(1) & 0XFF == 27: #escape
        break

