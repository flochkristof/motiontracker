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

from setuptools import find_packages, setup

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="MotionTrackerBeta",
    version="1.0.0",
    description='Motion Tracker Beta: A GUI based open-source motion tracking application',
    long_description=long_description,
    long_description_content_type="text/markdown",
    url='https://github.com/flochkristof/motiontracker',
    author='K. Floch', 
    author_email="kristof.floch@gmail.com",
    license='GPL-3.0',
    package_dir={"": "src"},
    packages=find_packages(where="src"),
    license_files = ('LICENSE.txt',),
    include_package_data=True,
    install_requires=['matplotlib==3.5.1',
        'scipy==1.8.0',
        'numpy==1.22.3',
        'opencv-contrib-python==4.5.5.64',
        #'pychebfun==0.3',
        'pynumdiff==0.1.2',
        'PyQt5==5.15.6',
       'cvxopt==1.3.0',
        'cvxpy==1.2.0',
        'pandas==1.4.2'],
    python_requires=">=3.4",
    classifiers=[
        'Development Status :: 5 - Production/Stable',
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: GNU General Public License v3 or later (GPLv3+)',  
        'Operating System :: POSIX :: Linux',
        'Operating System :: Microsoft :: Windows :: Windows 10',
        'Operating System :: MacOS',
        'Topic :: Scientific/Engineering :: Image Processing', 
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.10'
    ],
    entry_points={
        'console_scripts': [
            'MotionTrackerBeta=MotionTrackerBeta.main:MotionTracker',
            'MotionTracker=MotionTrackerBeta.main:MotionTracker'
        ]},

)
