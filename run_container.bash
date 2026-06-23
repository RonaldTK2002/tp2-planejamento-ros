docker run --name planner_container \
    -e TURTLEBOT3_MODEL=burger \
    --privileged -it --gpus all --network="host" \
    -v $PWD:/home/user/project/ \
    -v /tmp/.X11-unix:/tmp/.X11-unix \
    -e DISPLAY=$DISPLAY \
    -w /home/user/project/ \
    --volume="$HOME/.Xauthority:/root/.Xauthority:rw" \
    osrf/ros:noetic-desktop-full bash
