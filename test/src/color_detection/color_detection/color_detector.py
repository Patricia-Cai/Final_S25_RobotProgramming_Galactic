#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from std_msgs.msg import String
from cv_bridge import CvBridge
import cv2
import numpy as np

class ColorDetector(Node):
    def __init__(self):
        super().__init__('color_detector')
        
        # Initialize CV Bridge
        self.bridge = CvBridge()
        
        # Subscriber to camera image
        self.subscription = self.create_subscription(
            Image, 'image_raw', self.image_callback, 10)
        
        # Publisher for detected color
        self.color_pub = self.create_publisher(String, '/detected_color', 10)
        
        # Define color ranges in HSV (Hue, Saturation, Value)
        self.color_ranges = {
            'red': ([0, 100, 100], [10, 255, 255]),
            'green': ([40, 100, 100], [80, 255, 255]),
            'blue': ([100, 100, 100], [140, 255, 255]),
            'yellow': ([20, 100, 100], [40, 255, 255]),
        }
        
        self.get_logger().info("Color Detector Node Initialized")

    def image_callback(self, msg):
        try:
            # Convert ROS Image message to OpenCV image
            cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
            
            # Convert to HSV color space
            hsv_image = cv2.cvtColor(cv_image, cv2.COLOR_BGR2HSV)
            
            # Process the image to detect colors
            detected_color = self.detect_color(hsv_image)
            
            # Publish the detected color
            color_msg = String()
            color_msg.data = detected_color
            self.color_pub.publish(color_msg)
            
        except Exception as e:
            self.get_logger().error(f"Error processing image: {str(e)}")

    def detect_color(self, hsv_image):
        # Initialize variables
        max_pixels = 0
        detected_color = "unknown"
        
        # Blur the image to reduce noise
        blurred = cv2.GaussianBlur(hsv_image, (11, 11), 0)
        
        # Check each color range
        for color_name, (lower, upper) in self.color_ranges.items():
            # Create mask for the color range
            lower = np.array(lower, dtype=np.uint8)
            upper = np.array(upper, dtype=np.uint8)
            mask = cv2.inRange(blurred, lower, upper)
            
            # Apply morphological operations to remove small noise
            mask = cv2.erode(mask, None, iterations=2)
            mask = cv2.dilate(mask, None, iterations=2)
            
            # Count non-zero pixels in the mask
            pixel_count = cv2.countNonZero(mask)
            
            # Update detected color if this color has more pixels
            if pixel_count > max_pixels:
                max_pixels = pixel_count
                detected_color = color_name
        
        # Only return a color if enough pixels were detected
        if max_pixels > 1000:  # Adjust this threshold as needed
            return detected_color
        return "unknown"

def main(args=None):
    rclpy.init(args=args)
    color_detector = ColorDetector()
    rclpy.spin(color_detector)
    color_detector.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
