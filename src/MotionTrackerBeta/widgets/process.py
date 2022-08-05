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

import numpy as np

from MotionTrackerBeta.functions.helper import *
from MotionTrackerBeta.functions.differentiate import optimize_and_differentiate, differentiate

from MotionTrackerBeta.classes.classes import *


class PostProcesserThread(QThread):
    """Thread responsible for the post processing of the data collected by the trackers"""

    progressChanged = pyqtSignal(int)
    newObject = pyqtSignal(str)
    success = pyqtSignal()
    error_occured = pyqtSignal(str)

    def __init__(self, mode, objects_to_track, dt, parameters):
        """Initialization"""
        self.objects_to_track = objects_to_track
        self.dt = dt
        self.parameters = parameters
        self.progress = 0
        self.is_running = True
        self.mode = mode
        super(PostProcesserThread, self).__init__()

    def cancel(self):
        """Stops calculations exits the thread"""
        self.is_running = False

    def run(self):
        """Runs the post-processing code"""
        if self.mode:
            if self.parameters is None:
                self.error_occured.emit("Error: Invalid parameters!")
                return

            for i in range(len(self.objects_to_track)):
                M = self.objects_to_track[i]

                M.reset_output()

                x = np.array([p[0] for p in M.point_path])
                y = np.array([p[1] for p in M.point_path])

                # X coordinate
                if self.parameters[0]:
                    ret, xs, vx, ax = optimize_and_differentiate(
                        x, self.dt, self.parameters
                    )
                else:
                    ret, xs, vx, ax = differentiate(x, self.dt, self.parameters)

                if not ret:
                    if self.parameters[1]=="Sliding Chebychev Polynomial Fit":
                        self.error_occured.emit("Error: A porblem occured while calculating the derivative!\n-pychebfun is not installed.\nInstall it via pip install pychebfun!")
                        break
                    self.error_occured.emit(
                        "Error: A porblem occured while calculating the derivative!"
                    )
                    self.is_running = False
                    break
                self.progressChanged.emit(
                    int(100 / (2 * len(self.objects_to_track)) * (i + 1))
                )

                # Y coordinate
                if self.parameters[0]:
                    ret, ys, vy, ay = optimize_and_differentiate(
                        y, self.dt, self.parameters
                    )
                else:
                    ret, ys, vy, ay = differentiate(y, self.dt, self.parameters)

                if not ret:
                    if self.parameters[1]=="Sliding Chebychev Polynomial Fit":
                        self.error_occured.emit("Error: A porblem occured while calculating the derivative!\n-pychebfun is not installed.\nInstall it via pip install pychebfun!")
                        break
                    self.error_occured.emit(
                        "Error: A porblem occupred while calculating the derivative!"
                    )
                    self.is_running = False
                    break
                self.progressChanged.emit(
                    int(100 / len(self.objects_to_track) * (i + 1))
                )

                # Smoothed postion
                M.position = np.zeros((len(xs), 2))
                M.position[:, 0] = xs
                M.position[:, 1] = ys

                # Smoothed postion
                M.velocity = np.zeros((len(vx), 2))
                M.velocity[:, 0] = vx
                M.velocity[:, 1] = vy

                # Smoothed postion
                M.acceleration = np.zeros((len(ax), 2))
                M.acceleration[:, 0] = ax
                M.acceleration[:, 1] = ay
                self.success.emit()
        else:
            # here objetcts to track means only a single rotation object
            r = self.objects_to_track.rotation
            if self.parameters[0]:
                ret, rs, ang_v, ang_a = optimize_and_differentiate(
                    r, self.dt, self.parameters
                )
            else:
                ret, rs, ang_v, ang_a = differentiate(r, self.dt, self.parameters)
            if not ret:
                self.error_occured.emit(
                    "Error: A porblem occured while calculating the derivative!"
                )
                self.is_running = False
                return

            # Smoothed postion
            self.objects_to_track.rotation = rs
            self.objects_to_track.ang_velocity = ang_v
            self.objects_to_track.ang_acceleration = ang_a
            self.success.emit()