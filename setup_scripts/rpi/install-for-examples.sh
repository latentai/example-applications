#!/bin/bash

sudo bash ../add_latentai_debian_repository.sh
sudo apt update
sudo apt install -y cmake
sudo apt install -y libopencv-dev
sudo apt install -y latentai-runtime-cpu latentai-runtime-dev

# Installed in the lor setup script, not required here
# sudo apt install -y libjpeg-dev libopenblas-dev libopenmpi-dev libomp-dev

pip3 install opencv-python

