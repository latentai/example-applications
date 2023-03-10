# ******************************************************************************
# Copyright (c) 2019-2023 by Latent AI Inc. All Rights Reserved.
#
# This file is part of the example-applications (LRE) product,
# and is released under the Apache 2.0 License.
# *****************************************************************************/

#!/bin/bash

FLOAT32_MODEL=~/models/timm-gernet_m/x86_64_cuda/Float32-compile
INT8_MODEL=~/models/timm-gernet_m/x86_64_cuda/Int8-optimize

# Compile
mkdir build
cd build
cmake ..
make
cd ..


# FP32
mkdir -p $FLOAT32_MODEL/trt-cache/
TVM_TENSORRT_CACHE_DIR=$FLOAT32_MODEL/trt-cache/ ./build/bin/application $FLOAT32_MODEL/modelLibrary.so  ../../sample_images/apple.jpg ../../labels/class_names_10.txt

# FP16
mkdir -p $FLOAT32_MODEL/trt-cache/
TVM_TENSORRT_CACHE_DIR=$FLOAT32_MODEL/trt-cache/ TVM_TENSORRT_USE_FP16=1 ./build/bin/application $FLOAT32_MODEL/modelLibrary.so  ../../sample_images/apple.jpg ../../labels/class_names_10.txt

# INT8
mkdir -p $INT8_MODEL/trt-cache/
TVM_TENSORRT_CACHE_DIR=$INT8_MODEL/trt-cache/ TVM_TENSORRT_USE_INT8=1 TRT_INT8_PATH=$INT8_MODEL/.activations/ ./build/bin/application $FLOAT32_MODEL/modelLibrary.so  ../../sample_images/apple.jpg ../../labels/class_names_10.txt