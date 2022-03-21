#!/bin/bash

image=${1:-'nvcr.io/nvidia/pytorch:21.02-py3'}

cmd="docker run -it \
    -v $HOME/:$HOME \
    --expose=10000-11000 \
    -e DISPLAY=$DISPLAY -v /tmp/.X11-unix:/tmp/.X11-unix \
    --gpus all \
    --net=host --uts=host \
    --ipc=host --ulimit stack=67108864 --ulimit memlock=-1 --security-opt seccomp=unconfined  \
    $image \
    bash"

echo $cmd
eval $cmd

