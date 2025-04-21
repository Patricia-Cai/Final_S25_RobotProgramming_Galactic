# Final_S25_RobotProgramming_Galactic
Final project for S25 Intro to Robot Programming Course

## Team members
Zach Nobles

Jesse Kim

Joelle Wilensky

Jasmine King

Patricia Cai

Sriharsha Tangellamudi

Dyarra Mitchell


---

[opening the camera with python](http://www.yahboom.net/study/ROSMASTER-X3)

in a terminal run
```sh
ros2 launch astra_camera astro_pro_plus.launch.xml
```
```sh
ros2 launch astra_camera astra.launch.xml
```
build the workspace, source, and run
```sh
ros2 run colorcap webcam_pub
```
```sh
ros2 run colorcap webcam_sub
```
rqt_graph should show camera nodes
this python code reads the camera data
```python
#Import opecv library and cv_ Bridge Library
import cv2 as cv
from cv_bridge import CvBridge
#Creating CvBridge Objects
self.bridge = CvBridge()
#Define a subscriber to subscribe to RGB color image topic data published by deep camera nodes
self.sub_img =self.create_subscription(Image,'/camera/color/image_raw',self.handleTopic,100)
#Convert msg to image data, where bgr8 is the image encoding format
frame = self.bridge.imgmsg_to_cv2(msg, "bgr8")
```
