# ******************************************************************************
# Copyright (c) 2019-2023 by Latent AI Inc. All Rights Reserved.
#
# This file is part of the example-applications (LRE) product,
# and is released under the Apache 2.0 License.
# *****************************************************************************/

#!/bin/bash

FLOAT32_MODEL=$1
INT8_MODEL=$2
LABELS_PATH=../../labels/class_names_10.txt
IMAGE_PATH=../../sample_images/apple.jpg

if [ -v MODEL_PATH ]
then
    FLOAT32_MODEL=$MODEL_PATH/Float32-compile
    INT8_MODEL=$MODEL_PATH/Int8-optimize
fi
INT8_ACTIVATIONS=$INT8_MODEL/.activations/

# Compile
mkdir build
cd build
cmake ..
make
cd ..


echo "FP32..."
mkdir -p $FLOAT32_MODEL/trt-cache
TVM_TENSORRT_CACHE_DIR=$FLOAT32_MODEL/trt-cache/ ./build/bin/application $FLOAT32_MODEL/modelLibrary.so 10 $IMAGE_PATH $LABELS_PATH 

echo "FP16..."
mkdir -p $FLOAT32_MODEL/trt-cache
TVM_TENSORRT_CACHE_DIR=$FLOAT32_MODEL/trt-cache/ TVM_TENSORRT_USE_FP16=1 ./build/bin/application $FLOAT32_MODEL/modelLibrary.so 10 $IMAGE_PATH $LABELS_PATH

echo "INT8..."
mkdir -p $INT8_MODEL/trt-cache
TVM_TENSORRT_CACHE_DIR=$INT8_MODEL/trt-cache/ TVM_TENSORRT_USE_INT8=1 TRT_INT8_PATH=$INT8_ACTIVATIONS ./build/bin/application $INT8_MODEL/modelLibrary.so 10 $IMAGE_PATH $LABELS_PATH