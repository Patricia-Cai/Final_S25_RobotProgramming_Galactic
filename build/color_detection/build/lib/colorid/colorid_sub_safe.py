# I have absolutely no idea if any of this will work

from cv_bridge.core import CvBridgeError
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
from rclpy.node import Node
import numpy as np
import rclpy
import cv2
import sys

class ColoridSubscriber():
  def __init__(self):
    self.bridge = CvBridge()
    #Define a subscriber to subscribe to RGB color image topic data published by deep camera nodes
    self.sub_img = self.create_subscription(Image,'/camera/color/image_raw',self.handleTopic,100)
    #Convert msg to image data, where bgr8 is the image encoding format
    frame = self.bridge.imgmsg_to_cv2(msg, "bgr8")

  def getColor():
    # a more sophisticated algorithm should probably be used
    # but I want to see if this works first
    return "green"

def main(args=None):

    rclpy.init(args=args)
    subscriber = ColorSubscriber()
    rclpy.spin(subscriber)

    # Destroy the node explicitly
    # (optional - otherwise it will be done automatically
    # when the garbage collector destroys the node object)
    subscriber.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
