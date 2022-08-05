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

import numpy as np
import cv2
from MotionTrackerBeta.functions.transforms import *

def display_objects(
    frame,
    pos,
    section_start,
    section_stop,
    objects,
    box_bool,
    point_bool,
    trajectory_length,
):
    """Draws tracked object onto the video frame for playback and export"""
    for obj in objects:
        if obj.visible:
            if (pos >= section_start - 1) and (pos <= section_stop):
                # if pos == section_start - 1:
                #    if point_bool:
                #        x, y = obj.point
                #        frame = cv2.drawMarker(
                #            frame, (x, y), (0, 0, 255), 0, thickness=2
                #        )
                #    if box_bool:
                #        x0, y0, x1, y1 = tracker2gui(obj.rectangle)
                #        frame = cv2.rectangle(frame, (x0, y0), (x1, y1), (255, 0, 0), 2)
                #        cv2.putText(
                #            frame,
                #            obj.name,
                #            (x0, y0 - 5),
                #            cv2.FONT_HERSHEY_SIMPLEX,
                #            0.5,
                #            (255, 0, 0),
                #            1,
                #            cv2.LINE_AA,
                #        )
                # else:
                if point_bool:
                    x, y = obj.point_path[int(pos - section_start + 1)]
                    frame = cv2.drawMarker(
                        frame, (int(x), int(y)), (0, 0, 255), 0, thickness=2
                    )
                    if pos - section_start < trajectory_length:
                        for i in range(1, int(pos - section_start + 2)):
                            x0, y0 = obj.point_path[i - 1]
                            x1, y1 = obj.point_path[i]
                            # print(f"x:{x0}   y:{y0}   x:{x1}   y:{y1}")
                            frame = cv2.line(
                                frame,
                                (int(x0), int(y0)),
                                (int(x1), int(y1)),
                                (0, 0, 255),
                                2,
                            )
                    else:
                        for i in range(0, trajectory_length + +1):
                            x0, y0 = obj.point_path[
                                int(pos - section_start + i - trajectory_length)
                            ]
                            x1, y1 = obj.point_path[
                                int(pos - section_start + i - trajectory_length + 1)
                            ]
                            frame = cv2.line(
                                frame,
                                (int(x0), int(y0)),
                                (int(x1), int(y1)),
                                (0, 0, 255),
                                2,
                            )

                if box_bool:
                    x0, y0, x1, y1 = tracker2gui(
                        obj.rectangle_path[int(pos - section_start + 1)]
                    )
                    frame = cv2.rectangle(frame, (x0, y0), (x1, y1), (255, 0, 0), 2)
                    cv2.putText(
                        frame,
                        obj.name,
                        (x0, y0 - 5),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.5,
                        (255, 0, 0),
                        1,
                        cv2.LINE_AA,
                    )
    return frame

def draw_grid(x_num, y_num, frame, color_name):
    """Draws grid onto the frame with the given parameters"""
    dx = frame.shape[1] / x_num
    dy = frame.shape[0] / y_num

    # OpenCV uses BGR channels
    if color_name == "black":
        color = (0, 0, 0)
    elif color_name == "white":
        color = (255, 255, 255)
    elif color_name == "red":
        color = (0, 0, 255)
    elif color_name == "blue":
        color = (255, 0, 0)
    elif color_name == "green":
        color = (0, 255, 0)

    for x in np.linspace(start=dx, stop=frame.shape[1] - dx, num=x_num):
        x = int(round(x))
        cv2.line(
            frame, (x, 0), (x, frame.shape[0]), color=color, thickness=1,
        )

    for y in np.linspace(start=dy, stop=frame.shape[0] - dy, num=y_num):
        y = int(round(y))
        cv2.line(
            frame, (0, y), (frame.shape[1], y), color=color, thickness=1,
        )
    return frame