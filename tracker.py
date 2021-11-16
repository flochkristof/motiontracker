import cv2
import numpy as np
import math


class Motion:
    def __init__(self, **kwargs):
        self.camera = kwargs["camera"]
        self.fps = self.camera.get(cv2.CAP_PROP_FPS)
        self.status = "Motion initialized"

        # output
        self.timestamp = []
        self.rectangle_path = []
        self.point_path = []

    def track(self, start, stop, rectangle, tracker_type):
        # clear previous coordinates
        self.timestamp = []
        self.rectangle_path = []
        self.point_path = []

        # creating the tracker
        if tracker_type == "BOOSTING":
            tracker = cv2.TrackerBoosting_create()
        if tracker_type == "MIL":
            tracker = cv2.TrackerMIL_create()
        if tracker_type == "KCF":
            tracker = cv2.TrackerKCF_create()
        if tracker_type == "TLD":
            tracker = cv2.TrackerTLD_create()
        if tracker_type == "MEDIANFLOW":
            tracker = cv2.TrackerMedianFlow_create()
        if tracker_type == "GOTURN":
            tracker = cv2.TrackerGOTURN_create()
        if tracker_type == "MOSSE":
            tracker = cv2.TrackerMOSSE_create()
        if tracker_type == "CSRT":
            tracker = cv2.TrackerCSRT_create()

        # set camera to start frame, get the fps
        self.camera.set(cv2.CAP_PROP_POS_FRAMES, start)

        # initialize tracker
        ret, frame = self.camera.read()
        if not ret:
            self.status = "ERROR: Unable to read camera frame!"
            print("Unable to read camera frame!")
            return
        tracker.init(frame, rectangle)

        # tracking
        for i in range(stop - start):
            # read the next frame
            ret, frame = self.camera.read()
            if not ret:
                self.status = "ERROR: Unable to read camera frame!"
                print("Unable to read camera frame!")
                break

            # update the tracker
            ret, roi_box = tracker.update(frame)
            if ret:
                self.rectangle_path.append(roi_box)
                self.timestamp.append(i / self.fps)
                self.status = f"Tracking ... {math.ceil(i/(stop-start)*100)}"
                print(round(i / (stop - start) * 100))
            else:
                print("Calculation error!")


def select_rectangle(camera, start):
    camera.set(cv2.CAP_PROP_POS_FRAMES, start)
    # initialize tracker
    ret, frame = camera.read()
    if not ret:
        print("Unable to read camera frame!")
        return None
    box = cv2.selectROI("Select rectangle to track!", frame, False)
    return box


def select_point(camera, start):
    camera.set(cv2.CAP_PROP_POS_FRAMES, start)
    ret, frame = camera.read()
    if not ret:
        print("Unable to read camera frame!")
        return None

