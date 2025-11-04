# ArUco Marker Pose Estimation

This Python script uses OpenCV to detect ArUco markers (4x4, 5x5, 6x6) in real-time using the webcam. 
It estimates the pose (position and orientation) of each marker and overlays 3D axes on the video feed.
The marker positions (x, y, z in meters) are displayed on the screen.

Requirements:
- Python 3.7+
- opencv-contrib-python
- numpy

Run the script inside the `OpenCV` folder: 
`python aruco_pose.py`
