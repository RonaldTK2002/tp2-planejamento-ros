rodar run_container.bash
instalar  sudo apt update
sudo apt install ros-noetic-turtlebot3-simulations ros-noetic-map-server
source /opt/ros/noetic/setup.bash
catkin_make


roslaunch nossa_stack
roslaunch turtlebot3_slam turtlebot3_slam.launch slam_methods:=gmapping
roslaunch turtlebot3_teleop turtlebot3_teleop_key.launch
rosrun map_server map_saver -f ./meu_mapa