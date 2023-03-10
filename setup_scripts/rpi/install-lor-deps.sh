#!/bin/bash

sudo apt update -y
sudo apt upgrade -y
pip3 install Cython
pip3 install torch==1.13.0
pip3 install torchvision==0.14.1
pip3 install effdet

