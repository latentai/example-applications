# ******************************************************************************
# Copyright (c) 2019-2023 by Latent AI Inc. All Rights Reserved.
#
# This file is part of the example-applications (LRE) product,
# and is released under the Apache 2.0 License.
# *****************************************************************************/

#!/bin/bash

MODEL_PATH=<path to compiled artifacts>
IMAGE_PATH=../../sample_images/bus.jpg

model=MOBNETSSD # Detector architecture  supported YOLO, MOBNETSSD, EFFICIENTDET, NANODET

if [ -v MODEL_PATH ];
then
    echo "Models to be run from" $MODEL_PATH
    FLOAT32_MODEL=$MODEL_PATH/Float32-compile
    INT8_MODEL=$MODEL_PATH/Int8-optimize
fi
INT8_ACTIVATIONS=$INT8_MODEL/.activations/

# Set torch path for post processing
if [ -d ~/.torch-apps/libtorch ]
then
    TORCH_PATH=~/.torch-apps/libtorch
else
    TORCH_PATH=`python3 -c 'import torch;print(torch.utils.cmake_prefix_path)'`
fi
echo $TORCH_PATH

# Compile
mkdir build
cd build
cmake -DCMAKE_PREFIX_PATH=$TORCH_PATH ..
make -j 8
cd ..

echo "FP32..."
mkdir -p $FLOAT32_MODEL/trt-cache/
TVM_TENSORRT_CACHE_DIR=$FLOAT32_MODEL/trt-cache/ ./build/bin/application $FLOAT32_MODEL/modelLibrary.so 10 $IMAGE_PATH $model

echo "FP16..."
mkdir -p $FLOAT32_MODEL/trt-cache/
TVM_TENSORRT_CACHE_DIR=$FLOAT32_MODEL/trt-cache/ TVM_TENSORRT_USE_FP16=1 ./build/bin/application $FLOAT32_MODEL/modelLibrary.so 10 $IMAGE_PATH $model

echo "INT8..."
mkdir -p $INT8_MODEL/trt-cache/
TVM_TENSORRT_CACHE_DIR=$INT8_MODEL/trt-cache/ TVM_TENSORRT_USE_INT8=1 TRT_INT8_PATH=$INT8_ACTIVATIONS ./build/bin/application $INT8_MODEL/modelLibrary.so 10 $IMAGE_PATH $model