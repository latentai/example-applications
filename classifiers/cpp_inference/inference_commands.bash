# ******************************************************************************
# Copyright (c) 2019-2023 by Latent AI Inc. All Rights Reserved.
#
# This file is part of the example-applications (LRE) product,
# and is released under the Apache 2.0 License.
# *****************************************************************************/

#!/bin/bash

while [[ "$#" -gt 0 ]]; do
    case $1 in
        --model_path) MODEL_PATH="$2"; shift ;;
        --img_path) IMAGE_PATH="$2"; shift ;;
        --iterations) ITERATIONS="$2"; shift ;;
        --label_file) LABELS_PATH="$2"; shift ;;
        *) echo "Unknown parameter passed: $1"; exit 1 ;;
    esac
    shift
done

if [ -v MODEL_PATH ];
then
    echo "Models to be run from" $MODEL_PATH
    FLOAT32_MODEL=$MODEL_PATH/Float32-compile
    INT8_MODEL=$MODEL_PATH/Int8-optimize
fi

# Set Timing Cache for TensorRT Targets (Optional, Speeds up Engine Building)
current_path=$(pwd)

# Extract the desired part of the path
# This removes the '/detectors/cpp_inference' part from the end
desired_path="${current_path%/detectors/cpp_inference}"

export LAI_TENSORRT_TIMING_CACHE="$desired_path"

# Compile
mkdir build
cd build
cmake ..
make
cd ..


echo "FP32..."
mkdir -p $FLOAT32_MODEL/trt-cache
TVM_TENSORRT_CACHE_DIR=$FLOAT32_MODEL/trt-cache/ ./build/bin/application --precision float32 --model_path $FLOAT32_MODEL/modelLibrary.so --iterations 100 --img_path $IMAGE_PATH --label_file $LABELS_PATH 

echo "FP16..."
mkdir -p $FLOAT32_MODEL/trt-cache
TVM_TENSORRT_CACHE_DIR=$FLOAT32_MODEL/trt-cache/ ./build/bin/application --precision float16 --model_path $FLOAT32_MODEL/modelLibrary.so --iterations 100 --img_path $IMAGE_PATH --label_file $LABELS_PATH

echo "INT8..."
mkdir -p $INT8_MODEL/trt-cache
TVM_TENSORRT_CACHE_DIR=$INT8_MODEL/trt-cache/ ./build/bin/application --precision int8 --model_path $INT8_MODEL/modelLibrary.so --iterations 100 --img_path $IMAGE_PATH --label_file $LABELS_PATH