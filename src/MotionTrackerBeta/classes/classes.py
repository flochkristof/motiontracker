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


import math
import numpy as np


class Motion:
    """Class that stores every detail of the objects being tracked"""

    def __init__(self, name, point, rectangle, visible=True):
        # identification
        self.name = name

        # starting properties
        self.point = point
        self.rectangle = rectangle
        self.visible = visible

        # output, raw data
        self.rectangle_path = []
        self.point_path = []
        self.size_change = []

        # filtered, calculated output
        self.position = None
        self.velocity = None
        self.acceleration = None

    def __str__(self):
        return self.name

    def reset_data(self):
        self.rectangle_path = []
        self.point_path = []
        self.size_change = []

        # filtered, calculated output
        self.position = None
        self.velocity = None
        self.acceleration = None

    def reset_output(self):
        self.position = None
        self.velocity = None
        self.acceleration = None

    def can_plot(self):
        if (
            self.position is not None
            and self.acceleration is not None
            and self.velocity is not None
        ):
            return True
        return False


class Rotation:
    """Object that stores rotation data"""

    def __init__(self, P1, P2):
        self.P1 = P1
        self.P2 = P2
        self.rotation = None
        self.ang_velocity = None
        self.ang_acceleration = None

    def can_plot(self):
        if (
            self.rotation is not None
            and self.ang_velocity is not None
            and self.ang_acceleration is not None
        ):
            return True
        return False

    def calculate(self):
        P = np.vectorize(lambda P: P.point_path)

        path1 = P(self.P1)
        path2 = P(self.P2)
        rotation_init = np.arctan2(
            self.P2.point[1] - self.P1.point[1], self.P2.point[0] - self.P1.point[0]
        )
        self.rotation = -(
            np.arctan2(path2[:, 1] - path1[:, 1], path2[:, 0] - path1[:, 0])
            - rotation_init
        )

    def __str__(self):
        return self.P1.name + " - " + self.P2.name


class Ruler:
    """Class to define a milimeter scale on the video frames"""

    def __init__(self):
        self.x0 = None
        self.y0 = None
        self.x1 = None
        self.y1 = None
        self.mm = None
        self.mm_per_pix = None
        self.rdy = False
        self.visible = True

    def setP0(self, x, y):
        self.x0 = int(x)
        self.y0 = int(y)
        self.calculate()

    def setP1(self, x, y):
        self.x1 = int(x)
        self.y1 = int(y)
        self.calculate()

    def clear(self):
        self.x0 = None
        self.y0 = None
        self.x1 = None
        self.y1 = None
        self.rdy = False

    def calculate(self):
        if (
            self.x0 is not None
            and self.y0 is not None
            and self.x1 is not None
            and self.y1 is not None
            and self.mm is not None
        ):
            pix = math.sqrt(
                math.pow(self.x1 - self.x0, 2) + math.pow(self.y1 - self.y0, 2)
            )
            self.mm_per_pix = self.mm / pix
            self.rdy = True

    def displayable(self):
        if (
            self.x0 is not None
            and self.y0 is not None
            and self.x1 is not None
            and self.y1 is not None
        ):
            return True
        else:
            return False

    def reset(self):
        self.x0 = None
        self.y0 = None
        self.x1 = None
        self.y1 = None
        self.mm = None
        self.mm_per_pix = None
        self.rdy = False


class Logger:
    """Logger class"""

    def __init__(self):
        self.log = []

    def log(self, text):
        self.log.append(text)