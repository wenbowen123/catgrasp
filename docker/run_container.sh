docker rm -f catgrasp
CATGRASP_DIR=$(pwd)/../
xhost +  && docker run --gpus all --env NVIDIA_DISABLE_REQUIRE=1 -it --network=host --name catgrasp  --cap-add=SYS_PTRACE --security-opt seccomp=unconfined -v /home/bowen:/home/bowen -v $CATGRASP_DIR:/home/catgrasp -v /tmp/.X11-unix:/tmp/.X11-unix -v /tmp:/tmp  --ipc=host -e DISPLAY=${DISPLAY} -e GIT_INDEX_FILE wenbowen123/catgrasp bash