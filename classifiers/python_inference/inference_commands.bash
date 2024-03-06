# ******************************************************************************
# Copyright (c) 2019-2023 by Latent AI Inc. All Rights Reserved.
#
# This file is part of the example-applications (LRE) product,
# and is released under the Apache 2.0 License.
# *****************************************************************************/

#!/bin/bash

# Set vars from arguments
while [[ "$#" -gt 0 ]]; do
    case $1 in
        --model_path) MODEL_PATH="$2"; shift ;;
        --float32_model_binary_path) FLOAT32_MODEL="$2"; shift ;;
        --int8_model_binary_path) INT8_MODEL="$2"; shift ;;
        --input_image_path) IMAGE_PATH="$2"; shift ;;
        --labels_path) LABELS_PATH="$2"; shift ;;
        --iterations) ITERATIONS="$2"; shift ;; # Certain hardware targets dynamically optimize execution, these will need to run inference for a couple of iterations to see stable latency per inference.
        *) echo "Unknown parameter passed: $1"; exit 1 ;;
    esac
    shift
done

# Set default vars
if [ -z "$IMAGE_PATH" ];
then
    IMAGE_PATH=../../sample_images/apple.jpg
fi

if [ -z "$LABELS_PATH" ];
then
    LABELS_PATH=../../labels/class_names_10.txt
fi

if [ -z "$ITERATIONS" ];
then
    ITERATIONS=100
fi

if [ -v MODEL_PATH ];
then
    echo "Selecting models to run from" $MODEL_PATH
    FLOAT32_MODEL=$MODEL_PATH/Float32-compile
    INT8_MODEL=$MODEL_PATH/Int8-optimize
fi

if [ -v FLOAT32_MODEL ];
then
    echo "FP32..."
    mkdir -p $FLOAT32_MODEL/trt-cache
    TVM_TENSORRT_CACHE_DIR=$FLOAT32_MODEL/trt-cache python3 infer.py  \
        --precision float32  \
        --model_binary_path $FLOAT32_MODEL  \
        --input_image_path $IMAGE_PATH  \
        --labels $LABELS_PATH \
        --iterations $ITERATIONS

    echo "FP16..."
    mkdir -p $FLOAT32_MODEL/trt-cache
    TVM_TENSORRT_CACHE_DIR=$FLOAT32_MODEL/trt-cache python3 infer.py  \
        --precision float16 \
        --model_binary_path $FLOAT32_MODEL  \
        --input_image_path $IMAGE_PATH \
        --labels $LABELS_PATH \
        --iterations $ITERATIONS
fi

if [ -v INT8_MODEL ];
then
    echo "INT8..."
    mkdir -p $INT8_MODEL/trt-cache
    TVM_TENSORRT_CACHE_DIR=$INT8_MODEL/trt-cache python3 infer.py  \
        --precision int8  \
        --model_binary_path $INT8_MODEL \
        --input_image_path $IMAGE_PATH \
        --labels $LABELS_PATH \
        --iterations $ITERATIONS
fi