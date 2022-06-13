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
import cv2
import pynumdiff
import pynumdiff.optimize


def get_from_list_by_name(list, name):
    """Get object from list by name"""
    index = next((i for i, item in enumerate(list) if str(item) == name), -1,)
    if index != -1:
        return list[index]
    return None


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


def get_unit(parameters):
    """Returns the corresponding axis labels for the matplotlib plots"""

    if parameters["mode"] == "SIZ":
        return "Size change (%)"
    elif parameters["mode"] == "MOV":
        if parameters["unit"] == "m":
            if parameters["prop"] == "POS":
                return r"Position $\mathregular{(m)}$"
            elif parameters["prop"] == "VEL":
                return r"Velocity $\mathregular{(\frac{m}{s})}$"
            elif parameters["prop"] == "ACC":
                return r"Acceleration $\mathregular{(\frac{m}{s^{2}})}$"
        elif parameters["unit"] == "mm":
            if parameters["prop"] == "POS":
                return r"Position $\mathregular{(mm)}$"
            elif parameters["prop"] == "VEL":
                return r"Velocity $\mathregular{(\frac{mm}{s})}$"
            elif parameters["prop"] == "ACC":
                return r"Acceleration $\mathregular{(\frac{mm}{s^{2}})}$"
        elif parameters["unit"] == "pix":
            if parameters["prop"] == "POS":
                return r"Position $\mathregular{(pixel)}$"
            elif parameters["prop"] == "VEL":
                return r"Velocity $\mathregular{(\frac{pixel}{s})}$"
            elif parameters["prop"] == "ACC":
                return r"Acceleration $\mathregular{(\frac{pixel}{s^{2}})}$"
    elif parameters["mode"] == "ROT":
        if parameters["unit"] == "DEG":
            if parameters["prop"] == "POS":
                return r"Angular rotation $\mathregular{(^\circ)}$"
            elif parameters["prop"] == "VEL":
                return r"Angular velocity $\mathregular{(\frac{^\circ}{s})}$"
            elif parameters["prop"] == "ACC":
                return r"Angular acceleration $\mathregular{(\frac{^\circ}{s^{2}})}$"
        elif parameters["unit"] == "RAD":
            if parameters["prop"] == "POS":
                return r"Angular rotation $\mathregular{(rad)}$"
            elif parameters["prop"] == "VEL":
                return r"Angular velocity $\mathregular{(\frac{rad}{s})}$"
            elif parameters["prop"] == "ACC":
                return r"Angular acceleration $\mathregular{(\frac{rad}{s^{2}})}$"


def get_unit_readable(parameters):
    """Returns the corresponding axis labels for the matplotlib plots"""

    if parameters["mode"] == "SIZ":
        return "Size change (%)"
    elif parameters["mode"] == "MOV":
        if parameters["unit"] == "m":
            if parameters["prop"] == "POS":
                return "m"
            elif parameters["prop"] == "VEL":
                return "m/s"
            elif parameters["prop"] == "ACC":
                return "m/s^2"
        elif parameters["unit"] == "mm":
            if parameters["prop"] == "POS":
                return "mm"
            elif parameters["prop"] == "VEL":
                return "mm/s"
            elif parameters["prop"] == "ACC":
                return "mm/s^2"
        elif parameters["unit"] == "pix":
            if parameters["prop"] == "POS":
                return "pix"
            elif parameters["prop"] == "VEL":
                return "pix/s"
            elif parameters["prop"] == "ACC":
                return "pix/s^2"
    elif parameters["mode"] == "ROT":
        if parameters["unit"] == "DEG":
            if parameters["prop"] == "POS":
                return "deg"
            elif parameters["prop"] == "VEL":
                return "deg/s"
            elif parameters["prop"] == "ACC":
                return "deg/s^2"
        elif parameters["unit"] == "RAD":
            if parameters["prop"] == "POS":
                return "rad"
            elif parameters["prop"] == "VEL":
                return "rad/s"
            elif parameters["prop"] == "ACC":
                return "rad/s^2"


def rad2deg_(data):
    """Converts the data in the array from radian to edgrees"""

    if data.shape[1] > 1:
        data[:, [1, data.shape[1] - 1]] = 180 / np.pi * data[:, [1, data.shape[1] - 1]]
    else:
        data[:, 1] = data[:, 1] * 180 / np.pi
    return data


def pix2mm(data, pix_per_mm):
    """Using the ruler it coverts the data to mm"""

    if pix_per_mm is not None:
        if data.shape[1] > 1:
            data[:, [1, data.shape[1] - 1]] = (
                pix_per_mm * data[:, [1, data.shape[1] - 1]]
            )
        else:
            data[:, 1] = data[:, 1] * pix_per_mm
        return data
    else:
        return None


def pix2m(data, pix_per_mm):
    """Using the ruler it coverts the data to m"""

    if pix_per_mm is not None:
        if data.shape[1] > 1:
            data[:, [1, data.shape[1] - 1]] = (
                pix_per_mm / 1000 * data[:, [1, data.shape[1] - 1]]
            )
        else:
            data[:, 1] = data[:, 1] * pix_per_mm / 1000
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


def differentiate(p, dt, parameters):
    """
    Calculates the velocity and the acceleration based on the given position data using the selected algortithm
    
    :param p: (np.array of floats, 1xN) time series to differentiate
    :param dt: (float) time step
    :param parameters: (dict) parameters for the differentiation 
    :return: ret : returns True if the differentiation was successful
    :return: ps : smoothed position
    :return: v : calculated velocity
    :return: a : calculated acceleration
    """

    if parameters[1] == "First Order Finite Difference":
        try:
            ps, v = pynumdiff.finite_difference.first_order(p, dt)
            v, a = pynumdiff.finite_difference.first_order(v, dt)
            return True, ps, v, a
        except:
            return False, 0, 0, 0

    elif parameters[1] == "Second Order Finite Difference":
        try:
            ps, v = pynumdiff.finite_difference.second_order(p, dt)
            v, a = pynumdiff.finite_difference.second_order(v, dt)
            return True, ps, v, a
        except:
            return False, 0, 0, 0

    elif parameters[1] == "Iterated First Order Finite Difference":
        try:
            ps, v = pynumdiff.finite_difference.first_order(
                p, dt, parameters[2], parameters[3]
            )
            v, a = pynumdiff.finite_difference.first_order(
                v, dt, parameters[2], parameters[3]
            )
            return True, ps, v, a
        except:
            return False, 0, 0, 0
    elif parameters[1] == "Finite Difference with Median Smoothing":
        try:
            ps, v = pynumdiff.smooth_finite_difference.mediandiff(
                p, dt, parameters[2], parameters[3]
            )
            v, a = pynumdiff.smooth_finite_difference.mediandiff(
                v, dt, parameters[2], parameters[3]
            )
            return True, ps, v, a
        except Exception as e:
            print(e)
            return False, 0, 0, 0
    elif parameters[1] == "Finite Difference with Mean Smoothing":
        try:
            ps, v = pynumdiff.smooth_finite_difference.meandiff(
                p, dt, parameters[2], parameters[3]
            )
            v, a = pynumdiff.smooth_finite_difference.meandiff(
                v, dt, parameters[2], parameters[3]
            )
            return True, ps, v, a
        except Exception as e:
            print(e)
            return False, 0, 0, 0
    elif parameters[1] == "Finite Difference with Gaussian Smoothing":
        try:
            ps, v = pynumdiff.smooth_finite_difference.gaussiandiff(
                p, dt, parameters[2], parameters[3]
            )
            v, a = pynumdiff.smooth_finite_difference.gaussiandiff(
                v, dt, parameters[2], parameters[3]
            )
            return True, ps, v, a
        except Exception as e:
            print(e)
            return False, 0, 0, 0
    elif parameters[1] == "Finite Difference with Butterworth Smoothing":
        try:
            ps, v = pynumdiff.smooth_finite_difference.butterdiff(
                p, dt, parameters[2], parameters[3]
            )
            v, a = pynumdiff.smooth_finite_difference.butterdiff(
                v, dt, parameters[2], parameters[3]
            )
            return True, ps, v, a
        except Exception as e:
            print(e)
            return False, 0, 0, 0
    elif parameters[1] == "Finite Difference with Friedrichs Smoothing":
        try:
            ps, v = pynumdiff.smooth_finite_difference.friedrichsdiff(
                p, dt, parameters[2], parameters[3]
            )
            v, a = pynumdiff.smooth_finite_difference.friedrichsdiff(
                v, dt, parameters[2], parameters[3]
            )
            return True, ps, v, a
        except Exception as e:
            print(e)
            return False, 0, 0, 0
    elif parameters[1] == "Finite Difference with Spline Smoothing":
        try:
            ps, v = pynumdiff.smooth_finite_difference.splinediff(
                p, dt, parameters[2], parameters[3]
            )
            v, a = pynumdiff.smooth_finite_difference.splinediff(
                v, dt, parameters[2], parameters[3]
            )
            return True, ps, v, a
        except Exception as e:
            print(e)
            return False, 0, 0, 0
    elif (
        parameters[1]
        == "Iterative Total Variation Regularization with Regularized Velocity"
    ):
        try:
            parameters[3]["cg_maxiter"] = len(p)
            parameters[3]["scale"] = "small" if len(p) < 1000 else "large"
            ps, v = pynumdiff.total_variation_regularization.iterative_velocity(
                p, dt, parameters[2], parameters[3]
            )
            v, a = pynumdiff.total_variation_regularization.iterative_velocity(
                v, dt, parameters[2], parameters[3]
            )
            return True, ps, v, a
        except Exception as e:
            print(e)
            return False, 0, 0, 0
    elif (
        parameters[1]
        == "Convex Total Variation Regularization with Regularized Velocity"
    ):
        try:
            (
                ps,
                v,
            ) = pynumdiff.total_variation_regularization._total_variation_regularization.velocity(
                p, dt, parameters[2], parameters[3]
            )
            (
                v,
                a,
            ) = pynumdiff.total_variation_regularization._total_variation_regularization.velocity(
                v, dt, parameters[2], parameters[3]
            )
            return True, ps, v, a
        except Exception as e:
            print(e)
            return False, 0, 0, 0
    elif (
        parameters[1]
        == "Convex Total Variation Regularization with Regularized Acceleration"
    ):
        try:
            (
                ps,
                v,
            ) = pynumdiff.total_variation_regularization._total_variation_regularization.acceleration(
                p, dt, parameters[2], parameters[3]
            )
            (
                v,
                a,
            ) = pynumdiff.total_variation_regularization._total_variation_regularization.acceleration(
                v, dt, parameters[2], parameters[3]
            )
            return True, ps, v, a
        except Exception as e:
            print(e)
            return False, 0, 0, 0
    elif parameters[1] == "Convex Total Variation Regularization with Regularized Jerk":
        try:
            (
                ps,
                v,
            ) = pynumdiff.total_variation_regularization._total_variation_regularization.jerk(
                p, dt, parameters[2], parameters[3]
            )
            (
                v,
                a,
            ) = pynumdiff.total_variation_regularization._total_variation_regularization.jerk(
                v, dt, parameters[2], parameters[3]
            )
            return True, ps, v, a
        except Exception as e:
            print(e)
            return False, 0, 0, 0
    elif parameters[1] == "Convex Total Variation Regularization with Sliding Jerk":
        try:
            (
                ps,
                v,
            ) = pynumdiff.total_variation_regularization._total_variation_regularization.sliding_jerk(
                p, dt, parameters[2], parameters[3]
            )
            (
                v,
                a,
            ) = pynumdiff.total_variation_regularization._total_variation_regularization.sliding_jerk(
                v, dt, parameters[2], parameters[3]
            )
            return True, ps, v, a
        except Exception as e:
            print(e)
            return False, 0, 0, 0
    elif (
        parameters[1]
        == "Convex Total Variation Regularization with Smoothed Acceleration"
    ):
        try:
            (
                ps,
                v,
            ) = pynumdiff.total_variation_regularization._total_variation_regularization.smooth_acceleration(
                p, dt, parameters[2], parameters[3]
            )
            (
                v,
                a,
            ) = pynumdiff.total_variation_regularization._total_variation_regularization.smooth_acceleration(
                v, dt, parameters[2], parameters[3]
            )
            return True, ps, v, a
        except Exception as e:
            print(e)
            return False, 0, 0, 0
    elif parameters[1] == "Spectral Derivative":
        try:
            ps, v = pynumdiff.linear_model._linear_model.spectraldiff(
                p, dt, parameters[2], parameters[3]
            )
            v, a = pynumdiff.linear_model._linear_model.spectraldiff(
                v, dt, parameters[2], parameters[3]
            )
            return True, ps, v, a
        except Exception as e:
            print(e)
            return False, 0, 0, 0
    elif parameters[1] == "Savitzky-Golay Filter":
        try:
            ps, v = pynumdiff.linear_model._linear_model.savgoldiff(
                p, dt, parameters[2], parameters[3]
            )
            v, a = pynumdiff.linear_model._linear_model.savgoldiff(
                v, dt, parameters[2], parameters[3]
            )
            return True, ps, v, a
        except Exception as e:
            print(e)
            return False, 0, 0, 0
    elif parameters[1] == "Sliding Polynomial Derivative":
        try:
            ps, v = pynumdiff.linear_model._linear_model.polydiff(
                p, dt, parameters[2], parameters[3]
            )
            v, a = pynumdiff.linear_model._linear_model.polydiff(
                v, dt, parameters[2], parameters[3]
            )
            return True, ps, v, a
        except Exception as e:
            print(e)
            return False, 0, 0, 0
    elif parameters[1] == "Sliding Chebychev Polynomial Fit":
        try:
            ps, v = pynumdiff.linear_model._linear_model.chebydiff(
                p, dt, parameters[2], parameters[3]
            )
            v, a = pynumdiff.linear_model._linear_model.chebydiff(
                v, dt, parameters[2], parameters[3]
            )
            return True, ps, v, a
        except Exception as e:
            print(e)
            return False, 0, 0, 0
    else:
        return False, 0, 0, 0


def optimize_and_differentiate(p, dt, parameters):
    """Calculates the velocity and the acceleration based on the given position data using optimization to determine the ideal parameters of the delected differentiation algorithm"""
    gamma = np.exp(-1.6 * np.log(parameters[2]) - 0.71 * np.log(dt) - 5.1)

    if parameters[1] == "Iterated First Order Finite Difference":
        try:
            params, val = pynumdiff.optimize.finite_difference.first_order(
                p, dt, params=None, options={"iterate": True}, tvgamma=gamma
            )
            ps, v = pynumdiff.finite_difference.first_order(p, dt, params)
            params, val = pynumdiff.optimize.finite_difference.first_order(
                v, dt, params=None, options={"iterate": True}, tvgamma=gamma
            )
            v, a = pynumdiff.finite_difference.first_order(
                v, dt, params, options={"iterate": True}
            )
            return True, ps, v, a
        except Exception as e:
            print(e)
            return False, 0, 0, 0
    elif parameters[1] == "Finite Difference with Median Smoothing":
        try:
            params, val = pynumdiff.optimize.smooth_finite_difference.mediandiff(
                p, dt, params=None, options={"iterate": True}, tvgamma=gamma
            )
            ps, v = pynumdiff.smooth_finite_difference.mediandiff(p, dt, params)
            params, val = pynumdiff.optimize.finite_difference.first_order(
                v, dt, params=None, options={"iterate": True}, tvgamma=gamma
            )
            v, a = pynumdiff.smooth_finite_difference.mediandiff(v, dt, params)
            return True, ps, v, a
        except Exception as e:
            print(e)
            return False, 0, 0, 0
    elif parameters[1] == "Finite Difference with Mean Smoothing":
        try:
            params, val = pynumdiff.optimize.smooth_finite_difference.meandiff(
                p, dt, params=None, options={"iterate": True}, tvgamma=gamma
            )
            ps, v = pynumdiff.smooth_finite_difference.meandiff(p, dt, params)
            params, val = pynumdiff.optimize.smooth_finite_difference.meandiff(
                v, dt, params=None, options={"iterate": True}, tvgamma=gamma
            )
            v, a = pynumdiff.smooth_finite_difference.meandiff(v, dt, params)
            return True, ps, v, a
        except Exception as e:
            print(e)
            return False, 0, 0, 0
    elif parameters[1] == "Finite Difference with Gaussian Smoothing":
        try:
            params, val = pynumdiff.optimize.smooth_finite_difference.gaussiandiff(
                p, dt, params=None, options={"iterate": True}, tvgamma=gamma
            )
            ps, v = pynumdiff.smooth_finite_difference.gaussiandiff(p, dt, params)
            params, val = pynumdiff.optimize.smooth_finite_difference.gaussiandiff(
                v, dt, params=None, options={"iterate": True}, tvgamma=gamma
            )
            v, a = pynumdiff.smooth_finite_difference.gaussiandiff(v, dt, params)
            return True, ps, v, a
        except Exception as e:
            print(e)
            return False, 0, 0, 0
    elif parameters[1] == "Finite Difference with Butterworth Smoothing":
        try:
            params, val = pynumdiff.optimize.smooth_finite_difference.butterdiff(
                p, dt, params=None, options={"iterate": True}, tvgamma=gamma
            )
            ps, v = pynumdiff.smooth_finite_difference.butterdiff(p, dt, params)
            params, val = pynumdiff.optimize.smooth_finite_difference.butterdiff(
                v, dt, params=None, options={"iterate": True}, tvgamma=gamma
            )
            v, a = pynumdiff.smooth_finite_difference.butterdiff(v, dt, params)
            return True, ps, v, a
        except Exception as e:
            print(e)
            return False, 0, 0, 0
    elif parameters[1] == "Finite Difference with Friedrichs Smoothing":
        try:
            params, val = pynumdiff.optimize.smooth_finite_difference.friedrichsdiff(
                p, dt, params=None, options={"iterate": True}, tvgamma=gamma
            )
            ps, v = pynumdiff.smooth_finite_difference.friedrichsdiff(p, dt, params)
            params, val = pynumdiff.optimize.smooth_finite_difference.friedrichsdiff(
                p, dt, params=None, options={"iterate": True}, tvgamma=gamma
            )
            v, a = pynumdiff.smooth_finite_difference.friedrichsdiff(v, dt, params)
            return True, ps, v, a
        except Exception as e:
            print(e)
            return False, 0, 0, 0
    elif parameters[1] == "Finite Difference with Spline Smoothing":
        try:
            params, val = pynumdiff.optimize.smooth_finite_difference.splinediff(
                p, dt, params=None, options={"iterate": True}, tvgamma=gamma
            )
            ps, v = pynumdiff.smooth_finite_difference.splinediff(p, dt, params)
            params, val = pynumdiff.optimize.smooth_finite_difference.splinediff(
                v, dt, params=None, options={"iterate": True}, tvgamma=gamma
            )
            v, a = pynumdiff.smooth_finite_difference.splinediff(v, dt, params)
            return True, ps, v, a
        except Exception as e:
            print(e)
            return False, 0, 0, 0
    elif (
        parameters[1]
        == "Iterative Total Variation Regularization with Regularized Velocity"
    ):
        try:
            options = {"cg_maxiter": len(p)}
            options["scale"] = "small" if len(p) < 1000 else "large"
            (
                params,
                val,
            ) = pynumdiff.optimize.total_variation_regularization.iterative_velocity(
                p, dt, params=None, tvgamma=gamma, options=options
            )
            ps, v = pynumdiff.total_variation_regularization.iterative_velocity(
                p, dt, params, options
            )
            (
                params,
                val,
            ) = pynumdiff.optimize.total_variation_regularization.iterative_velocity(
                v, dt, params=None, tvgamma=gamma, options=options
            )
            v, a = pynumdiff.total_variation_regularization.iterative_velocity(
                v, dt, params, options
            )
            return True, ps, v, a
        except Exception as e:
            print(e)
            return False, 0, 0, 0
    elif (
        parameters[1]
        == "Convex Total Variation Regularization with Regularized Velocity"
    ):
        try:

            params, val = pynumdiff.optimize.total_variation_regularization.velocity(
                p, dt, params=None, tvgamma=gamma
            )
            (
                ps,
                v,
            ) = pynumdiff.total_variation_regularization._total_variation_regularization.velocity(
                p, dt, params
            )

            params, val = pynumdiff.optimize.total_variation_regularization.velocity(
                v, dt, params=None, tvgamma=gamma
            )
            (
                v,
                a,
            ) = pynumdiff.total_variation_regularization._total_variation_regularization.velocity(
                v, dt, params
            )
            return True, ps, v, a
        except Exception as e:
            print(e)
            return False, 0, 0, 0
    elif (
        parameters[1]
        == "Convex Total Variation Regularization with Regularized Acceleration"
    ):
        try:
            (
                params,
                val,
            ) = pynumdiff.optimize.total_variation_regularization.acceleration(
                p, dt, params=None, tvgamma=gamma
            )
            (
                ps,
                v,
            ) = pynumdiff.total_variation_regularization._total_variation_regularization.acceleration(
                p, dt, params
            )
            (
                params,
                val,
            ) = pynumdiff.optimize.total_variation_regularization.acceleration(
                v, dt, params=None, tvgamma=gamma
            )
            (
                v,
                a,
            ) = pynumdiff.total_variation_regularization._total_variation_regularization.acceleration(
                v, dt, params
            )
            return True, ps, v, a
        except Exception as e:
            print(e)
            return False, 0, 0, 0
    elif parameters[1] in {
        "Convex Total Variation Regularization with Regularized Jerk",
        "Convex Total Variation Regularization with Sliding Jerk",
    }:
        try:
            (params, val) = pynumdiff.optimize.total_variation_regularization.jerk(
                p, dt, params=None, tvgamma=gamma
            )
            (
                ps,
                v,
            ) = pynumdiff.total_variation_regularization._total_variation_regularization.jerk(
                p, dt, params
            )
            (params, val) = pynumdiff.optimize.total_variation_regularization.jerk(
                v, dt, params=None, tvgamma=gamma
            )
            (
                v,
                a,
            ) = pynumdiff.total_variation_regularization._total_variation_regularization.jerk(
                v, dt, params
            )
            return True, ps, v, a
        except Exception as e:
            print(e)
            return False, 0, 0, 0

    elif (
        parameters[1]
        == "Convex Total Variation Regularization with Smoothed Acceleration"
    ):
        try:
            (
                params,
                val,
            ) = pynumdiff.optimize.total_variation_regularization.smooth_acceleration(
                p, dt, params=None, tvgamma=gamma
            )
            (
                ps,
                v,
            ) = pynumdiff.total_variation_regularization._total_variation_regularization.smooth_acceleration(
                p, dt, params
            )
            (
                params,
                val,
            ) = pynumdiff.optimize.total_variation_regularization.smooth_acceleration(
                v, dt, params=None, tvgamma=gamma
            )
            (
                v,
                a,
            ) = pynumdiff.total_variation_regularization._total_variation_regularization.smooth_acceleration(
                v, dt, params
            )
            return True, ps, v, a
        except Exception as e:
            print(e)
            return False, 0, 0, 0
    elif parameters[1] == "Spectral Derivative":
        try:
            params, val = pynumdiff.optimize.linear_model.spectraldiff(
                p, dt, params=None, tvgamma=gamma
            )

            ps, v = pynumdiff.linear_model._linear_model.spectraldiff(p, dt, params)

            params, val = pynumdiff.optimize.linear_model.spectraldiff(
                v, dt, params=None, tvgamma=gamma
            )

            v, a = pynumdiff.linear_model._linear_model.spectraldiff(v, dt, params)
            return True, ps, v, a
        except Exception as e:
            print(e)
            return False, 0, 0, 0
    elif parameters[1] == "Savitzky-Golay Filter":
        try:
            params, val = pynumdiff.optimize.linear_model.savgoldiff(
                p, dt, params=None, tvgamma=gamma
            )
            ps, v = pynumdiff.linear_model._linear_model.savgoldiff(p, dt, params)
            params, val = pynumdiff.optimize.linear_model.savgoldiff(
                v, dt, params=None, tvgamma=gamma
            )
            v, a = pynumdiff.linear_model._linear_model.savgoldiff(v, dt, params)
            return True, ps, v, a
        except Exception as e:
            print(e)
            return False, 0, 0, 0
    elif parameters[1] == "Sliding Polynomial Derivative":
        try:
            params, val = pynumdiff.optimize.linear_model.polydiff(
                p, dt, params=None, tvgamma=gamma
            )
            ps, v = pynumdiff.linear_model._linear_model.polydiff(p, dt, params)
            params, val = pynumdiff.optimize.linear_model.polydiff(
                v, dt, params=None, tvgamma=gamma
            )
            v, a = pynumdiff.linear_model._linear_model.polydiff(v, dt, params)
            return True, ps, v, a
        except Exception as e:
            print(e)
            return False, 0, 0, 0
    elif parameters[1] == "Sliding Chebychev Polynomial Fit":
        try:
            params, val = pynumdiff.optimize.linear_model.chebydiff(
                p, dt, params=None, tvgamma=gamma
            )
            ps, v = pynumdiff.linear_model._linear_model.chebydiff(p, dt, params)
            params, val = pynumdiff.optimize.linear_model.chebydiff(
                v, dt, params=None, tvgamma=gamma
            )
            v, a = pynumdiff.linear_model._linear_model.chebydiff(v, dt, params)
            return True, ps, v, a
        except Exception as e:
            print(e)
            return False, 0, 0, 0
    else:
        return False, 0, 0, 0
