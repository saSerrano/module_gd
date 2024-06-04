# Gesture Detection Basic Module
This is the ROS package of the Gesture Detection Basic Module. This module is in charge of recognizing non-verbal forms of language that people might use to complement their spoken commands to a robot.

## Installation
Create a conda environment using the requirments file ```gesture_det_env.yml``` and the following command:
```conda env create -f gesture_det_env.yml```

## Running the Module
Before running the module, make sure you first activate the conda environment:
```conda activate gesture_det_env```

To run the module in ROS Melodic and later versions use the following command:
```roslaunch module_gd bm.launch```

To run it in ROS Hydro, depending on the sensor configuration you wish to use, there are three options:
- Orbbec Astra RGBD sensor: ```roslaunch module_gd orbbec_roshydro.launch```
- Kinect-Azure RGBD sensor (FOV narrow mode): ```roslaunch module_gd kinect_narrow_roshydro.launch```
- Kinect-Azure RGBD sensor (FOV wide mode): ```roslaunch module_gd kinect_wide_roshydro.launch```

## Future Tasks
- **Further Testing**: Test it using the kinect-azure sensor (so far it has only been tested with the orbbec sensor).
- **Joints 3D Enhancement**: Stability of the joints' x-y-z coordinates ectraxtion should be improved.
- **Add time-out**: Add time-out feature for the detection of gestures. In this was the robot will not try forever to detect the requested gesture.
- **Add Gestures/Postures**: So  far the module only recognizes where the person is pointing at with both of its hands. Being able to recognize other gestures/postures (e.g. being sitted, layed at a sofa, raising the hand, saluting, etc.) would be a great improvement.
