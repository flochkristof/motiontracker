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


def get_from_list_by_name(list, name):
    """Get object from list by name"""
    index = next((i for i, item in enumerate(list) if str(item) == name), -1,)
    if index != -1:
        return list[index]
    return None


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