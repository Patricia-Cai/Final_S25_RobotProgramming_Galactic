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

def imgmsg_to_cv2(img_msg):
    if img_msg.encoding != "bgr8":
        logger = RcutilsLogger("my_logger")
        logger.error("This Coral detect node has been hardcoded to the 'bgr8' encoding.  Come change the code if you're actually trying to implement a new camera")

    dtype = np.dtype("uint8")
    dtype = dtype.newbyteorder('>' if img_msg.is_bigendian else '<')
    image_opencv = np.ndarray(shape=(img_msg.height, img_msg.width, 3),
                    dtype=dtype, buffer=img_msg.data)
    
    if img_msg.is_bigendian == (sys.byteorder == 'little'):
        image_opencv = image_opencv.byteswap().newbyteorder()
    
    return image_opencv

def cv2_to_imgmsg(cv_image):
    img_msg = Image()
    img_msg.height = cv_image.shape[0]
    img_msg.width = cv_image.shape[1]
    img_msg.encoding = "bgr8"
    img_msg.is_bigendian = 0
    img_msg.data = cv_image.tostring()
    img_msg.step = len(img_msg.data) // img_msg.height
    return img_msg

class WebcamSub(Node):
    def __init__(self):
        super().__init__('stream_node')

        self.bridge = CvBridge()
        self.image_counter = 0
        self.current_image = None
        
        # Create directory for saved images if it doesn't exist
        self.save_dir = "captured_images"
        os.makedirs(self.save_dir, exist_ok=True)

        # define subscriber
        self.img_subscription = self.create_subscription(Image, 'image_raw', self.img_callback, 1)
        self.img_subscription  # prevent unused variable warning

        # Create a window and set mouse callback
        cv2.namedWindow("Camera Feed")
        cv2.setWindowTitle("Camera Feed", "Press 'c' to capture - 'q' to quit")

    def img_callback(self, img_msg):
        try:
            self.current_image = self.bridge.imgmsg_to_cv2(img_msg, 'bgr8')
        except CvBridgeError as e:
            self.get_logger().info(e)
        
        if self.current_image is not None:
            # Show live feed
            display_image = self.current_image.copy()
            
            # Add instruction text
            cv2.putText(display_image, "Press 'c' to capture", (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
            cv2.putText(display_image, "Press 'q' to quit", (10, 70), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
            
            cv2.imshow("Camera Feed", display_image)
            
            # Check for key press
            key = cv2.waitKey(1) & 0xFF
            if key == ord('c'):
                self.capture_and_process()
            elif key == ord('q'):
                self.cleanup_and_shutdown()

    def capture_and_process(self):
        if self.current_image is None:
            return
            
        processed_image, prominent_hsv = self.process_image(self.current_image)
        
        # Save the image
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        filename = f"{self.save_dir}/capture_{timestamp}_{self.image_counter}.png"
        cv2.imwrite(filename, processed_image)
        self.get_logger().info(f"Image captured: {filename}")
        self.get_logger().info(f"Prominent HSV: {prominent_hsv}")
        self.image_counter += 1
        
        # Show processed image until key press
        cv2.imshow("Processed Image", processed_image)
        cv2.waitKey(0)  # Wait indefinitely for any key press
        cv2.destroyWindow("Processed Image")

    def process_image(self, image):
        # Convert to HSV color space
        hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        
        # Find most prominent color in HSV
        hsv_pixels = hsv_image.reshape(-1, 3)
        hsv_counts = defaultdict(int)
        for pixel in hsv_pixels:
            # Quantize hue to 30 degrees, saturation/value to 50 for grouping
            h = pixel[0] // 30 * 30
            s = pixel[1] // 50 * 50
            v = pixel[2] // 50 * 50
            hsv_counts[(h, s, v)] += 1
        prominent_hsv = max(hsv_counts.items(), key=lambda x: x[1])[0]
        
        # Create output image with overlays
        output = image.copy()
        height, width = output.shape[:2]
        
        # Create overlay background
        overlay = np.zeros((150, width, 3), dtype=np.uint8)
        
        # Add HSV information
        hsv_text = f"HSV: {prominent_hsv}"
        cv2.putText(overlay, hsv_text, (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        
        # Create color block showing the HSV color
        hsv_color_block = np.zeros((70, width-20, 3), dtype=np.uint8)
        hsv_color_block[:,:] = cv2.cvtColor(np.array([[prominent_hsv]], dtype=np.uint8), cv2.COLOR_HSV2BGR)[0,0]
        overlay[60:130, 10:width-10] = hsv_color_block
        
        # Combine with original image
        output = np.vstack([output, overlay])
        
        return output, prominent_hsv

    def cleanup_and_shutdown(self):
        cv2.destroyAllWindows()
        self.destroy_node()
        rclpy.shutdown()
        sys.exit(0)

def main(args=None):
    rclpy.init(args=args)
    imgsub_obj = WebcamSub()
    rclpy.spin(imgsub_obj)
    imgsub_obj.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()