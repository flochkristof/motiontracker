<p align="center">
  <img src="https://user-images.githubusercontent.com/65981382/166214135-47ecd327-cba8-47c0-a034-9f6f14b777ce.png" alt="Motion Tracker Beta"/>
</p>

# Motion Tracker Beta
An easy-to-use, standalone and open source motion tracking application aimed at researchers and engineers, written in Python.

## Features
- Intutitive graphical user interface
- Capable of handling the most common video formats
- Capable of tracking various properties of multiple objects simultaneously
- Diverse set of built in tracking algorithms, based on the `OpenCV` libary
- Rich selection of numerical differentiation algorithms powered by the `PyNumDiff` libary
- Built in plotting an exporting features


For the complete list of features please check the [documentation](docs/DOCUMENTATION.pdf).

## Dependencies
The Graphical user interface was created with the [PyQt5](https://www.riverbankcomputing.com/software/pyqt/) framework. For the handling of video files and to do the actual tracking the [OpenCV](https://opencv.org/) library was used with its built in tracking algorithms. Numerical differentiations are carried out using the [PyNumDiff](https://github.com/florisvb/PyNumDiff). Plots and figures are generated by [matplotlib](https://matplotlib.org/). For the complete list of required packages check [requirements.txt](src/requirements.txt)

## Installation
### Download source 
- Make sure you have `Python 3.8.3` or newer installed
- Download and extract files
- cd into `src` folder
- Install required dependencies
```
$ pip install -r requirements.txt
```
- Run `main_app.py`
```
$ python main_app.py
```
### Download binaries
- Download binaries extract it to your specified location
- Open application with `Motion Tracker Beta.exe`
### Download the installer
- Download and run the installer
- Follow the given instructions
- After successfull installation the software is accessible under the name `Motion Tracker Beta`
# Usage
For a detailed guide about the software check out the documentation.
# License
Motion Tracker Beta is released under the `GNU General Public License v3.0`.
# Author
The software was developed by Kristof Floch at the Department of Applied Mechanics, Faculty of Mechanical Engineering, Budapest University of Technology and Economics.
### Contact
- E-mail: kristof.floch@gmail.com
