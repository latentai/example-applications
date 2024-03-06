# ******************************************************************************
# Copyright (c) 2019-2023 by Latent AI Inc. All Rights Reserved.
#
# This file is part of the example-applications (LRE) product,
# and is released under the Apache 2.0 License.
# *****************************************************************************/

#!/bin/bash


while [[ "$#" -gt 0 ]]; do
    case $1 in
        --model_path) MODEL_PATH="$2"; shift ;;  # supported detectors: YOLO, MOBNETSSD, EFFICIENTDET, NANODET
        --img_path) IMAGE_PATH="$2"; shift ;;
        --iterations) ITERATIONS="$2"; shift ;;
        --model_family) MODEL_FAMILY="$2"; shift ;;
        --conf_thres) CONFIDENCE_THRESHOLD="$2"; shift ;; # A higher confidence threshold speeds up post processing but may skip some needed detections
        --iou_thres) IOU_THRESHOLD="$2"; shift ;;
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

# Set torch path for post processing
if [ -d ~/.torch-apps/libtorch ]
then
    TORCH_PATH=~/.torch-apps/libtorch
else
    TORCH_PATH=`python3 -c 'import torch;print(torch.utils.cmake_prefix_path)'`
fi
echo $TORCH_PATH

# Set Timing Cache for TensorRT Targets (Optional, Speeds up Engine Building)
current_path=$(pwd)

# Extract the desired part of the path
# This removes the '/detectors/cpp_inference' part from the end
desired_path="${current_path%/detectors/cpp_inference}"

export LAI_TENSORRT_TIMING_CACHE="$desired_path"

# Compile
mkdir build
cd build
cmake -DCMAKE_PREFIX_PATH=$TORCH_PATH ..
make -j$(nproc)
cd ..

echo "FP32..."
mkdir -p $FLOAT32_MODEL/trt-cache/
TVM_TENSORRT_CACHE_DIR=$FLOAT32_MODEL/trt-cache/ ./build/bin/application \
    --precision float32  \
    --model_path $FLOAT32_MODEL/modelLibrary.so  \
    --iterations $ITERATIONS  \
    --img_path $IMAGE_PATH  \
    --model_family $MODEL_FAMILY  \
    --iou_thres $IOU_THRESHOLD  \
    --conf_thres $CONFIDENCE_THRESHOLD

echo "FP16..."
mkdir -p $FLOAT32_MODEL/trt-cache/
TVM_TENSORRT_CACHE_DIR=$FLOAT32_MODEL/trt-cache/ ./build/bin/application \
    --precision float16  \
    --model_path $FLOAT32_MODEL/modelLibrary.so  \
    --iterations $ITERATIONS  \
    --img_path $IMAGE_PATH \
    --model_family $MODEL_FAMILY  \
    --iou_thres $IOU_THRESHOLD  \
    --conf_thres $CONFIDENCE_THRESHOLD

echo "INT8..."
mkdir -p $INT8_MODEL/trt-cache/
TVM_TENSORRT_CACHE_DIR=$INT8_MODEL/trt-cache/ ./build/bin/application \
    --precision int8  \
    --model_path $INT8_MODEL/modelLibrary.so  \
    --iterations $ITERATIONS  \
    --img_path $IMAGE_PATH  \
    --model_family $MODEL_FAMILY  \
    --iou_thres $IOU_THRESHOLD  \
    --conf_thres $CONFIDENCE_THRESHOLD
