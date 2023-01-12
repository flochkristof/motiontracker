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

from math import ceil
import numpy as np


def rad2deg_(data):
    """Converts the data in the array from radian to edgrees"""

    if data.shape[1] > 1:
        data[:, [1, data.shape[1] - 1]] = 180 / np.pi * data[:, [1, data.shape[1] - 1]]
    else:
        data[:, 1] = data[:, 1] * 180 / np.pi
    return data


def pix2mm(data, mm_per_pix):
    """Using the ruler it coverts the data to mm"""

    if mm_per_pix is not None:
        if data.shape[1] > 1:
            data[:, [1, data.shape[1] - 1]] = (
                mm_per_pix * data[:, [1, data.shape[1] - 1]]
            )
        else:
            data[:, 1] = data[:, 1] * mm_per_pix
        return data
    else:
        return None


def pix2m(data, mm_per_pix):
    """Using the ruler it coverts the data to m"""

    if mm_per_pix is not None:
        if data.shape[1] > 1:
            data[:, [1, data.shape[1] - 1]] = (
                mm_per_pix/1000 * data[:, [1, data.shape[1] - 1]]
            )
        else:
            data[:, 1] = data[:, 1] * mm_per_pix/1000
        return data
    else:
        return None


def list2np(data):
    """Converts the list of coordinates to np.array"""

    x = np.asarray([p[0] for p in data])
    y = np.asarray([p[1] for p in data])
    result = np.zeros((len(x), 2))
    result[:, 0] = x
    result[:, 1] = y
    return result


def crop_frame(frame, x_offset, y_offset, zoom):
    """Crops the frame according to offset and zoom parameters"""

    x0 = ceil(frame.shape[1] / 2)
    y0 = ceil(frame.shape[0] / 2)
    return frame[
        (y0 + y_offset - round(y0 * zoom)) : (y0 + y_offset + round(y0 * zoom)),
        (x0 + x_offset - round(x0 * zoom)) : (x0 + x_offset + round(x0 * zoom)),
    ]


def crop_roi(frame, rect):
    """Crop frame with the ROI rectangle"""
    return frame[rect[1] : rect[3], rect[0] : rect[2]]


def rect2cropped(rectangle, roi_rect):
    """Convert rectangle data to cropped coordinates"""
    x, y, w, h = rectangle
    return (x - roi_rect[0], y - roi_rect[1], w, h)


def gui2tracker(rectangle):
    """Converts the rectangle representation from the gui to the tracker (x0,y0,x1,y1)->(x,y,w,h)"""
    x1, y1, x2, y2 = rectangle
    return (min(x1, x2), min(y1, y2), abs(x2 - x1), abs(y2 - y1))


def tracker2gui(rectangle):
    """Converts the rectangle representation from the tracker to the gui (x,y,w,h)->(x0,y0,x1,y1)"""
    x, y, w, h = rectangle
    return (int(x), int(y), int(x + w), int(y + h))