# ******************************************************************************
# Copyright (c) 2019-2023 by Latent AI Inc. All Rights Reserved.
#
# This file is part of the example-applications (LRE) product,
# and is released under the Apache 2.0 License.
# *****************************************************************************/

#!/bin/bash

ARCH=aarch64_cuda_xavier_jp4
PATH_TO_DETECTOR_MODELS=/home/dev/models/recipe_hub

models=(NANODET MOBNETSSD EFFICIENTDET YOLO YOLO) # supported detector models
paths=($PATH_TO_DETECTOR_MODELS/nanodet/$ARCH
       $PATH_TO_DETECTOR_MODELS/ssdn/$ARCH
       $PATH_TO_DETECTOR_MODELS/efficientdetn/$ARCH
       $PATH_TO_DETECTOR_MODELS/yolov5n/$ARCH
       $PATH_TO_DETECTOR_MODELS/yolov8/$ARCH)

# Current known issue: These applications require a different libtorch
# (libtorch-cxx11-abi-shared-with-deps-2.0.1%2Bcu118.zip) installed that
# differs from the SDK version of libtorch.  The workaround is to install
# this version in the home directory for linking with the example
# applications when running in the SDK docker container.  This is the
# reason for the following workaround:

if [ -d ~/.torch-apps/libtorch ]
then
    TORCH_PATH=~/.torch-apps/libtorch
else
    TORCH_PATH=`python3 -c 'import torch;print(torch.utils.cmake_prefix_path)'`
fi
echo $TORCH_PATH

sed -i "s/#define $model .*/#define $model 1/"  include/processors.hpp

# Compile
mkdir build
cd build
cmake -DCMAKE_PREFIX_PATH=$TORCH_PATH ..
make -j 8
cd ..

# FP32
mkdir -p $FLOAT32_MODEL/trt-cache/
TVM_TENSORRT_CACHE_DIR=$FLOAT32_MODEL/trt-cache/ ./build/bin/application $FLOAT32_MODEL/modelLibrary.so 10 ../../sample_images/bus.jpg

# FP16
mkdir -p $FLOAT32_MODEL/trt-cache/
TVM_TENSORRT_CACHE_DIR=$FLOAT32_MODEL/trt-cache/ TVM_TENSORRT_USE_FP16=1 ./build/bin/application $FLOAT32_MODEL/modelLibrary.so 10 ../../sample_images/bus.jpg

# INT8
mkdir -p $INT8_MODEL/trt-cache/
TVM_TENSORRT_CACHE_DIR=$INT8_MODEL/trt-cache/ TVM_TENSORRT_USE_INT8=1 TRT_INT8_PATH=$INT8_MODEL/.activations/ ./build/bin/application $INT8_MODEL/modelLibrary.so 10 ../../sample_images/bus.jpg