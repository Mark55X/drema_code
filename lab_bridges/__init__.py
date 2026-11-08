"""
TCP bridges between the laboratory hardware and the DREMA lab environment (drema/client/lab).

The servers run on the robot PC, outside any container: ROS2 (rclpy) for joints, commands and ROS camera topics, the
ZED SDK for ZED cameras. The clients need only socket, msgpack and numpy. The camera and joint bridges speak the
protocol of Drema2's zed_camera / ros2_camera / ros2_joints, so a bridge already running for Drema2 serves this client
too. Each bridge starts with `python -m lab_bridges.<module> --help`.
"""
