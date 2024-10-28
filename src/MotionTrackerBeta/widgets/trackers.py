# Copyright 2022 Kristof Floch
 
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
# 
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
# 
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>.


from PyQt5.QtCore import QThread, pyqtSignal
import math
import cv2
from MotionTrackerBeta.functions.helper import *
from MotionTrackerBeta.functions.transforms import *

from MotionTrackerBeta.classes.classes import *

class TrackingThread(QThread):
    """Thread responsible for running the tracking algorithms"""

    # create signals
    progressChanged = pyqtSignal(int)
    newObject = pyqtSignal(str)
    success = pyqtSignal()
    rotation_calculated = pyqtSignal(Rotation)
    error_occured = pyqtSignal(str)

    def __init__(
        self,
        objects_to_track,
        camera,
        start,
        stop,
        tracker_type,
        size,
        fps,
        timestamp,
        roi_rect,
    ):
        """Intitialization"""
        self.objects_to_track = objects_to_track
        self.camera = camera
        self.section_start = start
        self.section_stop = stop
        self.tracker_type = tracker_type
        self.timestamp = timestamp
        self.roi_rect = roi_rect
        self.size = size
        self.fps = fps
        self.progress = "0"
        self.is_running = True
        if roi_rect is None:
            self.roi_rect = (0, 0)
        else:
            self.roi_rect = roi_rect

        # call parent function
        super(TrackingThread, self).__init__()

    def cancel(self):
        """Stops the """
        self.is_running = False

    def run(self):
        self.newObject.emit("Tracking objects...")

        # t0 = time.time()
        for j in range(len(self.objects_to_track)):
            M = self.objects_to_track[j]

            # emit the name of the tracked object
            self.newObject.emit("Tracking object: " + M.name + "...")

            # reset previous data
            M.reset_data()
            if j == 0:
                self.timestamp.clear()

            # creating the tracker
            if self.tracker_type == "BOOSTING":
                tracker = cv2.legacy.TrackerBoosting_create()
            if self.tracker_type == "MIL":
                tracker = cv2.legacy.TrackerMIL_create()
            if self.tracker_type == "KCF":
                tracker = cv2.TrackerKCF_create()
            if self.tracker_type == "TLD":
                tracker = cv2.legacy.TrackerTLD_create()
            if self.tracker_type == "MEDIANFLOW":
                tracker = cv2.legacy.TrackerMedianFlow_create()
            # if self.tracker_type == "GOTURN":
            #    tracker = cv2.TrackerGOTURN_create()
            if self.tracker_type == "MOSSE":
                tracker = cv2.legacy.TrackerMOSSE_create()
            if self.tracker_type == "CSRT":
                tracker = cv2.TrackerCSRT_create()

            # set camera to start frame, get the fps
            self.camera.set(cv2.CAP_PROP_POS_FRAMES, self.section_start - 1)

            # initialize tracker
            ret, frame = self.camera.read()
            if not ret:
                self.error_occured.emit("Unable to read video frame!")
                return

            # crop roi if provided
            if len(self.roi_rect) == 4:
                frame = crop_roi(frame, self.roi_rect)

            # initialize cv2 tracker
            try:
                tracker.init(frame, rect2cropped(M.rectangle, self.roi_rect))
            except:
                # handle error, stop the thread emit signal
                self.is_running = False
                self.error_occured.emit("Unable to initialize the tracker!")
                break

            # add to path list
            M.rectangle_path.append(M.rectangle)
            M.point_path.append(M.point)

            # record timestamp only for the first tracking
            if j == 0:
                self.timestamp.append(0)

            # for the calculation of the point
            dy = (M.point[1] - M.rectangle[1]) / M.rectangle[3]
            dx = (M.point[0] - M.rectangle[0]) / M.rectangle[2]

            # for zoom
            w0 = M.rectangle[2]
            h0 = M.rectangle[3]

            # tracking
            for i in range(int(self.section_stop - self.section_start)):
                # read the next frame
                ret, frame = self.camera.read()

                # handle errors
                if not ret:
                    self.error_occured.emit("Unable to read video frame!")
                    self.is_running = False
                    break

                # crop if roi provided
                if len(self.roi_rect) == 4:
                    frame = crop_roi(frame, self.roi_rect)

                # update the tracker
                try:
                    ret, roi_box = tracker.update(frame)
                except Exception as e:
                    self.error_occured.emit(f"Tracking failed!\n{e}")
                    self.is_running = False
                    break

                if ret:  # tracking duccessfull

                    # traditional tracking
                    x, y, w, h = roi_box
                    M.rectangle_path.append(
                        (self.roi_rect[0] + x, self.roi_rect[1] + y, w, h)
                    )

                    # M.rectangle_path.append(roi_box)
                    M.point_path.append(
                        (
                            self.roi_rect[0] + roi_box[0] + dx * roi_box[2],
                            self.roi_rect[1] + roi_box[1] + dy * roi_box[3],
                        )
                    )

                    # change of size
                    if self.size:
                        M.size_change.append((roi_box[2] / w0 + roi_box[3] / h0) / 2)

                    # log timestamps
                    if j == 0:
                        self.timestamp.append((i + 1) / self.fps)

                    # progress
                    self.progress = math.ceil(
                        i / (self.section_stop - self.section_start) * 100
                    )
                    self.progressChanged.emit(self.progress)
                else:
                    # handle errors
                    self.error_occured.emit(
                        "Tracking failed!\n Tracker returned with failure!"
                    )
                    self.is_running = False

                if not self.is_running:
                    # reset path
                    M.rectangle_path = []
                    break

            if not self.is_running:
                break
        # set camera pos to start
        self.camera.set(cv2.CAP_PROP_POS_FRAMES, self.section_start)

        # emit success signal
        if self.is_running:
            # print(f"Finished in {time.time()-t0}")
            self.success.emit()


class TrackingThreadV2(QThread):
    """Thread responsible for running the tracking algorithms"""

    # create signals
    progressChanged = pyqtSignal(int)
    newObject = pyqtSignal(str)
    success = pyqtSignal()
    rotation_calculated = pyqtSignal(Rotation)
    error_occured = pyqtSignal(str)

    def __init__(
        self,
        objects_to_track,
        camera,
        start,
        stop,
        tracker_type,
        size,
        fps,
        timestamp,
        roi_rect,
    ):
        """Intitialization"""
        self.objects_to_track = objects_to_track
        self.camera = camera
        self.section_start = start
        self.section_stop = stop
        self.tracker_type = tracker_type
        self.timestamp = timestamp
        self.roi_rect = roi_rect
        self.size = size
        self.fps = fps
        self.progress = "0"
        self.is_running = True
        if roi_rect is None:
            self.roi_rect = (0, 0)
        else:
            self.roi_rect = roi_rect # in x0, y1, x1, y1 format

        # call parent function
        super(TrackingThreadV2, self).__init__()

    def cancel(self):
        """Stops the tracking"""
        self.is_running = False

    def run(self):
        # t0 = time.time()
        self.newObject.emit("Tracking objects...")

        # reset previous data
        self.timestamp.clear()

        # set camera to start frame, read it
        self.camera.set(cv2.CAP_PROP_POS_FRAMES, self.section_start - 1)

        # initialize tracker
        ret, frame = self.camera.read()

        # check errors
        if not ret:
            self.error_occured.emit("Unable to read video frame!")
            return

        # crop roi if provided
        if len(self.roi_rect) == 4:
            frame = crop_roi(frame, self.roi_rect)

        # creating tracker objects, adding them into list
        trackers = []
        for M in self.objects_to_track:
            # reset data
            M.reset_data()

            # creating the tracker
            if self.tracker_type == "BOOSTING":
                tracker = cv2.legacy.TrackerBoosting_create()
            if self.tracker_type == "MIL":
                tracker = cv2.legacy.TrackerMIL_create()
            if self.tracker_type == "KCF":
                tracker = cv2.TrackerKCF_create()
            if self.tracker_type == "TLD":
                tracker = cv2.legacy.TrackerTLD_create()
            if self.tracker_type == "MEDIANFLOW":
                tracker = cv2.legacy.TrackerMedianFlow_create()
            # if self.tracker_type == "GOTURN":
            #    tracker = cv2.TrackerGOTURN_create()
            if self.tracker_type == "MOSSE":
                tracker = cv2.legacy.TrackerMOSSE_create()
            if self.tracker_type == "CSRT":
                tracker = cv2.TrackerCSRT_create()

            # initialize cv2 with starting frame
            try:
                tracker.init(frame, rect2cropped(M.rectangle, self.roi_rect))
            except:
                # handle error, stop the thread emit signal
                self.is_running = False
                self.error_occured.emit("Unable to initialize the tracker!")
                return

            # store data
            M.rectangle_path.append(M.rectangle)
            M.point_path.append(M.point)

            # for zoom
            w0 = M.rectangle[2]
            h0 = M.rectangle[3]

            # size change if required
            if self.size:
                M.size_change.append((M.rectangle[2] / w0 + M.rectangle[3] / h0) / 2)

            # add tracker to list
            trackers.append(tracker)

        # store timestamp
        self.timestamp.append(0)

        # tracking loop
        for j in range(int(self.section_stop - self.section_start)):
            # read frame
            ret, frame = self.camera.read()

            # check for errors
            if not ret:
                return

            # crop roi if provided
            if len(self.roi_rect) == 4:
                frame = crop_roi(frame, self.roi_rect)

            # update trackers
            for i in range(len(trackers)):
                try:
                    ret, roi_box = trackers[i].update(frame)
                except Exception as e:
                    self.error_occured.emit(f"Tracking failed!\n{e}")
                    self.is_running = False
                    return

                # successfull tracking
                if ret:
                    # traditional tracking
                    x, y, w, h = roi_box

                    # for the calculation of the point
                    dy = (M.point[1] - M.rectangle[1]) / M.rectangle[3]
                    dx = (M.point[0] - M.rectangle[0]) / M.rectangle[2]

                    # for zoom
                    w0 = M.rectangle[2]
                    h0 = M.rectangle[3]

                    self.objects_to_track[i].rectangle_path.append(
                        (self.roi_rect[0] + x, self.roi_rect[1] + y, w, h)
                    )
                    # M.rectangle_path.append(roi_box)
                    self.objects_to_track[i].point_path.append(
                        (
                            self.roi_rect[0] + roi_box[0] + dx * roi_box[2],
                            self.roi_rect[1] + roi_box[1] + dy * roi_box[3],
                        )
                    )
                    # change of size
                    if self.size:
                        self.objects_to_track[i].size_change.append(
                            (roi_box[2] / w0 + roi_box[3] / h0) / 2
                        )
                else:
                    # handle errors
                    self.error_occured.emit(
                        "Tracking failed!\n Tracker returned with failure!"
                    )
                    self.is_running = False
                    return
                if not self.is_running:
                    break

            # timestamp
            self.timestamp.append((j + 1) / self.fps)

            # progress
            self.progress = math.ceil(
                j / (self.section_stop - self.section_start) * 100
            )
            self.progressChanged.emit(self.progress)

            # break loop if cancelled
            if not self.is_running:
                break

        # set camera position to start
        self.camera.set(cv2.CAP_PROP_POS_FRAMES, self.section_start)

        # emit success signal
        if self.is_running:
            # print(f"Finished in {time.time()-t0}")
            self.success.emit()
