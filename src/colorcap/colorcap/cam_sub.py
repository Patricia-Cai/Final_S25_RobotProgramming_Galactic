import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
import cv2
from cv_bridge import CvBridge
import numpy as np
import os
import time
from collections import defaultdict
from Rosmaster_Lib import Rosmaster
import sys

class ColorTrackingNode(Node):
    def __init__(self):
        super().__init__('color_tracking_node')
        self.bridge = CvBridge()
        self.bot = Rosmaster()  
        self.image_counter = 0
        self.current_image = None
        self.last_color = None  
        
        # Create directory for saved images
        self.save_dir = "captured_images"
        os.makedirs(self.save_dir, exist_ok=True)

        # Create subscriber
        self.img_subscription = self.create_subscription(Image, 'image_raw', self.img_callback, 1)
        
        # Create display window
        cv2.namedWindow("Camera Controls", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("Camera Controls", 640, 240)
        self.update_instructions()

    def update_instructions(self, last_capture=None):
        control_panel = np.zeros((240, 640, 3), dtype=np.uint8)

        cv2.putText(control_panel, "CONTROLS:", (20, 40), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
        cv2.putText(control_panel, "Press 'C' - CAPTURE and set lights", (20, 80), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.putText(control_panel, "Press 'R' - RESET lights", (20, 120), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 165, 0), 2)  
        cv2.putText(control_panel, "Press 'Q' - QUIT program", (20, 160), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
       
        if self.last_color:
            r, g, b = self.last_color
            cv2.putText(control_panel, f"Last RGB: ({r},{g},{b})", (20, 200, 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        
        if last_capture:
            cv2.putText(control_panel, f"Last capture: {os.path.basename(last_capture)}", (300, 200), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        
        cv2.imshow("Camera Controls", control_panel)

    def img_callback(self, img_msg):
        try:
            self.current_image = self.bridge.imgmsg_to_cv2(img_msg, 'bgr8')
        except Exception as e:
            self.get_logger().error(f"Image conversion error: {str(e)}")
            return
        
        if self.current_image is not None:
            # Show live feed
            cv2.imshow("Live Camera Feed", self.current_image)
            
            # Check for key presses
            key = cv2.waitKey(1) & 0xFF
            if key == ord('c') or key == ord('C'):
                self.capture_and_process()
            elif key == ord('r') or key == ord('R'):
                self.reset_lights()
            elif key == ord('q') or key == ord('Q'):
                self.cleanup_and_shutdown()

    def reset_lights(self):
        for led in range(1, 11):
            self.bot.set_colorful_lamps(led, 0, 0, 0)
        self.last_color = None
        self.get_logger().info("All lights reset")
        self.update_instructions() 
    def capture_and_process(self):
        if self.current_image is None:
            return
            
        # Process image to find dominant HSV color
        processed_image, prominent_hsv = self.process_image(self.current_image)
        
        # Convert HSV to BGR for the light bar
        hsv_color = np.uint8([[list(prominent_hsv)]])
        bgr_color = cv2.cvtColor(hsv_color, cv2.COLOR_HSV2BGR)[0][0]
        r, g, b = int(bgr_color[2]), int(bgr_color[1]), int(bgr_color[0])
        self.last_color = (r, g, b)  
        
        for led in range(1, 11):
            self.bot.set_colorful_lamps(led, r, g, b)
        
        # Save the image
        filename = os.path.join(self.save_dir, f"capture_{self.image_counter}.png")
        cv2.imwrite(filename, processed_image)
        
        # Update UI with feedback
        self.update_instructions(filename)
        self.get_logger().info(f"Image captured: {filename}")
        self.get_logger().info(f"Dominant HSV: {prominent_hsv} | RGB: ({r},{g},{b})")
        self.image_counter += 1
        
        # Show processed image
        cv2.imshow("Captured Image (Press any key to close)", processed_image)
        cv2.waitKey(0)
        cv2.destroyWindow("Captured Image (Press any key to close)")

    def process_image(self, image):
        hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        hsv_pixels = hsv_image.reshape(-1, 3)
        hsv_counts = defaultdict(int)
        
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
        self.reset_lights()
        cv2.destroyAllWindows()
        self.destroy_node()
        rclpy.shutdown()
        sys.exit(0)

def main(args=None):
    rclpy.init(args=args)
    node = ColorTrackingNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
