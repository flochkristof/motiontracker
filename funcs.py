from math import ceil
from matplotlib.pyplot import yscale
import numpy as np
from scipy.ndimage import gaussian_filter1d
from scipy.signal import savgol_filter
import cv2


def get_from_list_by_name(list, name):
    index = next((i for i, item in enumerate(list) if str(item) == name), -1,)
    if index != -1:
        return list[index]
    return None


def display_objects(frame, pos, section_start, section_stop, objects):
    """Draws tracked object onto the video frame for playback and export"""
    for obj in objects:
        if obj.visible:
            if (pos >= section_start - 1) and (pos <= section_stop):
                if pos == section_start - 1:
                    x, y = obj.point
                    frame = cv2.drawMarker(frame, (x, y), (0, 0, 255), 0, thickness=2)
                    x0, y0, x1, y1 = tracker2gui(obj.rectangle)
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
                else:
                    x, y = obj.point_path[int(pos - section_start)]
                    frame = cv2.drawMarker(
                        frame, (int(x), int(y)), (0, 0, 255), 0, thickness=2
                    )
                    x, y = (
                        obj.position[int(pos - section_start)][0],
                        obj.position[int(pos - section_start)][1],
                    )
                    frame = cv2.drawMarker(
                        frame, (int(x), int(y)), (0, 255, 255), 0, thickness=2,
                    )
                    x0, y0, x1, y1 = tracker2gui(
                        obj.rectangle_path[int(pos - section_start)]
                    )
                    # print(f"{x0} {y0} {x1} {y1}")
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
                return r"Position $\mathregular{(\frac{pixel}{s})}$"
            elif parameters["prop"] == "ACC":
                return r"Acceleration $\mathregular{(\frac{pixel}{s^{2}})}$"
    elif parameters["mode"] == "ROT":
        if parameters["unit"] == "DEG":
            if parameters["prop"] == "POS":
                return r"Position $\mathregular{(^\circ)}$"
            elif parameters["prop"] == "VEL":
                return r"Position $\mathregular{(\frac{^\circ}{s})}$"
            elif parameters["prop"] == "ACC":
                return r"Acceleration $\mathregular{(\frac{^\circ}{s^{2}})}$"
        elif parameters["unit"] == "RAD":
            if parameters["prop"] == "POS":
                return r"Position $\mathregular{(rad)}$"
            elif parameters["prop"] == "VEL":
                return r"Position $\mathregular{(\frac{rad}{s})}$"
            elif parameters["prop"] == "ACC":
                return r"Acceleration $\mathregular{(\frac{rad}{s^{2}})}$"


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


# def time_data_join(timestamp,)


def gaussian(data, window, sigma):
    """Gaussian filter"""

    # truncate
    t = (((window - 1) / 2) - 0.5) / sigma

    x = np.asarray([p[0] for p in data])
    y = np.asarray([p[1] for p in data])

    x = gaussian_filter1d(x, sigma=sigma, truncate=t, mode="nearest")
    y = gaussian_filter1d(y, sigma=sigma, truncate=t, mode="nearest")

    result = np.zeros((len(x), 2))

    result[:, 0] = x
    result[:, 1] = y
    return result


def moving_avg(data, w_len):
    """Mooving average filter"""
    window = np.ones(w_len) / w_len
    x = np.asarray([p[0] for p in data])
    y = np.asarray([p[1] for p in data])
    x = np.convolve(data[:, 0], window, mode="valid")
    y = np.convolve(data[:, 1], window, mode="valid")
    result = np.zeros((len(x), 2))
    result[:, 0] = x
    result[:, 1] = y


def savgol(data, window, pol):
    """Savitzky-Golay filter"""
    x = np.asarray([p[0] for p in data])
    y = np.asarray([p[1] for p in data])
    xs = savgol_filter(x, window, pol)
    ys = savgol_filter(y, window, pol)
    result = np.zeros((len(x), 2))
    result[:, 0] = xs
    result[:, 1] = ys
    return result


def savgol_diff(data, window, pol, order, fps=1):
    if window % 2 == 0:
        window += 1
    print(data)
    x = data[:, 0]  # .T
    y = data[:, 1]  # .transpose()
    print(x)
    dx = savgol_filter(x, window, pol, deriv=order) * (fps) ** order
    print(dx)
    dy = savgol_filter(y, window, pol, deriv=order) * (fps) ** order
    result = np.zeros((len(x), 2))

    result[:, 0] = dx
    result[:, 1] = dy
    return result


"""

    def derivate_SG(self, data, window, poly_n, order, time_scale):
        '''Savitzky-Golay filterrel való numerikus deriválás'''
        if isinstance(data[0], float) or isinstance(data[0], int):
            return signal.savgol_filter(data, window, poly_n, deriv=order)*(self.fps*time_scale)**order

        elif len(data[0]) == 2 or isinstance(data[0], (list, tuple)):
            result = np.array(data)
            x = result[:,0]
            y = result[:,1]
            dx_dt = signal.savgol_filter(x, window, poly_n, deriv=order)*(self.fps*time_scale)**order
            dy_dt = signal.savgol_filter(y, window, poly_n, deriv=order)*(self.fps*time_scale)**order
            result[:,0], result[:,1] = dx_dt, dy_dt
            return result
def derivative_FinDiff(self, data, accuracy, order, time_scale):
        '''Véges differencia módszerrel való deriválás tetszőleges renddel'''
    step = 1/(self.fps*time_scale)
    d_dt = FinDiff(0, step, order, acc=accuracy)
    if isinstance(data[0], float) or isinstance(data[0], int):
            return d_dt(data)
    elif len(data[0]) == 2 or isinstance(data[0], (list, tuple)):
            result = np.array(data)
        x = result[:,0]
        y = result[:,1]
        result[:,0], result[:,1] = d_dt(x), d_dt(y)
        return result
"""


def crop_frame(frame, x_offset, y_offset, zoom):
    """Crops the frame according to offset and zoom parameters"""

    x0 = ceil(frame.shape[1] / 2)
    y0 = ceil(frame.shape[0] / 2)
    return frame[
        (y0 + y_offset - round(y0 * zoom)) : (y0 + y_offset + round(y0 * zoom)),
        (x0 + x_offset - round(x0 * zoom)) : (x0 + x_offset + round(x0 * zoom)),
    ]


def crop_roi(frame, rect):
    return frame[
        rect[1] : rect[3], rect[0] : rect[2]
    ]  # TODO tracking alatt hozzáadni az eredményekhez a cuccot


def rect2cropped(rectangle, roi_rect):
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
