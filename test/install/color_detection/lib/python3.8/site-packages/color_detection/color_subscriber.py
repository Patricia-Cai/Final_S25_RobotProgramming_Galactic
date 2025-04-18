#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from std_msgs.msg import String

class ColorSubscriber(Node):
    def __init__(self):
        super().__init__('color_subscriber')
        
        # Subscriber to detected color
        self.subscription = self.create_subscription(
            String,
            '/detected_color',
            self.color_callback,
            10)
        
        self.get_logger().info("Color Subscriber Node Initialized")

    def color_callback(self, msg):
        self.get_logger().info(f"Detected color: {msg.data}")

def main(args=None):
    rclpy.init(args=args)
    color_subscriber = ColorSubscriber()
    rclpy.spin(color_subscriber)
    color_subscriber.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
