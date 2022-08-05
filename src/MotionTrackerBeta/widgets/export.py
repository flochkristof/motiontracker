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
import cv2
from MotionTrackerBeta.functions.helper import *
from MotionTrackerBeta.functions.display import display_objects
from MotionTrackerBeta.classes.classes import *


MODE=True # disables/ enables the optimization based differentiation methods
# True: python environment
# False: complied exe


class ExportingThread(QThread):
    """Thread responsible for exporting video with tracked objects displayed"""

    # create signals
    progressChanged = pyqtSignal(int)
    success = pyqtSignal()
    error_occured = pyqtSignal(str)

    def __init__(
        self,
        camera,
        objects_to_track,
        start,
        stop,
        filename,
        fps,
        box_bool,
        point_bool,
        trajectory_lenght,
    ):
        """Initialization"""
        self.camera = camera
        self.objects = objects_to_track
        self.section_start = start
        self.section_stop = stop
        self.filename = filename
        self.fps = fps
        self.is_running = True
        self.box_bool = box_bool
        self.point_bool = point_bool
        self.trajectory_length = trajectory_lenght

        # call parent function
        super(ExportingThread, self).__init__()

    def cancel(self):
        """Stops the calculation, exits the thread"""
        self.is_running = False

    def run(self):
        """Runs the exporting algorithm algoritm"""

        # initialize writer
        h = int(self.camera.get(cv2.CAP_PROP_FRAME_HEIGHT))
        w = int(self.camera.get(cv2.CAP_PROP_FRAME_WIDTH))
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(self.filename, fourcc, self.fps, (w, h))

        # goto start
        self.camera.set(cv2.CAP_PROP_POS_FRAMES, self.section_start)

        # export frame by frame
        for i in range(int(self.section_stop - self.section_start)):

            # get position
            pos = self.camera.get(cv2.CAP_PROP_POS_FRAMES)

            # read frame
            ret, frame = self.camera.read()

            # stop in case of error
            if not ret:
                self.is_running = False
                self.error_occured.emit("Unable to read video frame!")
                break

            # display objects
            frame = display_objects(
                frame,
                pos,
                self.section_start,
                self.section_stop,
                self.objects,
                self.box_bool,
                self.point_bool,
                self.trajectory_length,
            )

            # write frame to file
            writer.write(frame)

            # update progress
            self.progressChanged.emit(
                int(pos / (self.section_stop - self.section_start) * 100)
            )

        if self.is_running:
            # emit signal
            self.success.emit()
