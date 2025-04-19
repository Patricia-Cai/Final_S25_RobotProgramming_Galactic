#####
import rclpy
#####
from rclpy.impl.rcutils_logger import RcutilsLogger
from rclpy.node import Node

#####
from sensor_msgs.msg import Image
#####

import cv2
from cv_bridge import CvBridge
from cv_bridge.core import CvBridgeError
import numpy as np
import sys
import os
import time
from collections import defaultdict

class WebcamSub(Node):
    def __init__(self):
        super().__init__('stream_node')
        self.bridge = CvBridge()
        self.image_counter = 0
        self.current_image = None
        self.last_key_press_time = 0
        self.key_press_cooldown = 1.0  # seconds
        
        # Create directory for saved images
        self.save_dir = "captured_images"
        os.makedirs(self.save_dir, exist_ok=True)

        # Create subscriber
        self.img_subscription = self.create_subscription(
            Image, 'image_raw', self.img_callback, 1)
        
        # Create display window with clear instructions
        cv2.namedWindow("Camera Controls", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("Camera Controls", 640, 480)
        self.update_instructions()

    def update_instructions(self, last_capture=None):
        """Create a clear control panel with visual feedback"""
        control_panel = np.zeros((200, 640, 3), dtype=np.uint8)
        
        # Main instructions
        cv2.putText(control_panel, "CONTROLS:", (20, 40), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
        cv2.putText(control_panel, "Press 'C' to CAPTURE current frame", (20, 80), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.putText(control_panel, "Press 'Q' to QUIT", (20, 120), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        
        # Last capture feedback
        if last_capture:
            cv2.putText(control_panel, f"Last capture: {last_capture}", (20, 160), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        
        cv2.imshow("Camera Controls", control_panel)

    def img_callback(self, img_msg):
        try:
            self.current_image = self.bridge.imgmsg_to_cv2(img_msg, 'bgr8')
        except CvBridgeError as e:
            self.get_logger().info(e)
        
        if self.current_image is not None:
            # Show live feed in a separate window
            cv2.imshow("Live Camera Feed", self.current_image)
            
            # Check for key presses with cooldown
            current_time = time.time()
            if current_time - self.last_key_press_time > self.key_press_cooldown:
                key = cv2.waitKey(1) & 0xFF
                if key == ord('c') or key == ord('C'):
                    self.last_key_press_time = current_time
                    self.capture_and_process()
                elif key == ord('q') or key == ord('Q'):
                    self.cleanup_and_shutdown()

    def capture_and_process(self):
        if self.current_image is None:
            return
            
        # Process image to find dominant HSV color
        processed_image, prominent_hsv = self.process_image(self.current_image)
        
        # Save the image
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        filename = f"{self.save_dir}/capture_{timestamp}_{self.image_counter}.png"
        cv2.imwrite(filename, processed_image)
        
        # Update UI with feedback
        self.update_instructions(filename)
        self.get_logger().info(f"Image captured: {filename}")
        self.get_logger().info(f"Dominant HSV: {prominent_hsv}")
        self.image_counter += 1
        
        # Show processed image
        cv2.imshow("Captured Image (Press any key to close)", processed_image)
        cv2.waitKey(0)
        cv2.destroyWindow("Captured Image (Press any key to close)")

    def process_image(self, image):
        hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        hsv_pixels = hsv_image.reshape(-1, 3)
        hsv_counts = defaultdict(int)
        
        # Quantize HSV values
        for pixel in hsv_pixels:
            h = pixel[0] // 30 * 30
            s = pixel[1] // 50 * 50
            v = pixel[2] // 50 * 50
            hsv_counts[(h, s, v)] += 1
        
        prominent_hsv = max(hsv_counts.items(), key=lambda x: x[1])[0]
        
        # Create output with overlay
        output = image.copy()
        overlay = np.zeros((150, image.shape[1], 3), dtype=np.uint8)
        
        # Add HSV info
        cv2.putText(overlay, f"Dominant HSV: {prominent_hsv}", (20, 40), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        # Show color sample
        color_sample = np.zeros((70, 200, 3), dtype=np.uint8)
        color_sample[:,:] = cv2.cvtColor(np.array([[prominent_hsv]], dtype=np.uint8), cv2.COLOR_HSV2BGR)[0,0]
        overlay[60:130, 20:220] = color_sample
        
        return np.vstack([output, overlay]), prominent_hsv

    def cleanup_and_shutdown(self):
        cv2.destroyAllWindows()
        self.destroy_node()
        rclpy.shutdown()
        sys.exit(0)

def main(args=None):
    rclpy.init(args=args)
    webcam_sub = WebcamSub()
    rclpy.spin(webcam_sub)
    webcam_sub.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
