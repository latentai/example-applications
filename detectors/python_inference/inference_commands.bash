# ******************************************************************************
# Copyright (c) 2019-2023 by Latent AI Inc. All Rights Reserved.
#
# This file is part of the example-applications (LRE) product,
# and is released under the Apache 2.0 License.
# *****************************************************************************/

#!/bin/bash

FLOAT32_MODEL=$1
INT8_MODEL=$2
LABELS_PATH=$3
FORMAT=$4
MAX_DET=$5
CONF_THRESHOLD=$6
IOU_THRESHOLD=$7

IMAGE_PATH=../../sample_images/bus.jpg
REPRESENTATION=$8

if [ -v MODEL_PATH ];
then
    FLOAT32_MODEL=$MODEL_PATH/Float32-compile
    INT8_MODEL=$MODEL_PATH/Int8-optimize
fi
INT8_ACTIVATIONS=$INT8_MODEL/.activations/ # can we throw an error if this does not exist?

echo "FP32..."
mkdir -p $FLOAT32_MODEL/trt-cache
TVM_TENSORRT_CACHE_DIR=$FLOAT32_MODEL/trt-cache python3 infer.py --path_to_model $FLOAT32_MODEL  --input_image $IMAGE_PATH  --labels $LABELS_PATH --format $FORMAT --representation $REPRESENTATION --max_det $MAX_DET --confidence $CONF_THRESHOLD --iou $IOU_THRESHOLD

echo "FP16..."
mkdir -p $FLOAT32_MODEL/trt-cache
TVM_TENSORRT_CACHE_DIR=$FLOAT32_MODEL/trt-cache TVM_TENSORRT_USE_FP16=1 python3 infer.py --path_to_model $FLOAT32_MODEL  --input_image $IMAGE_PATH --labels $LABELS_PATH --format $FORMAT --representation $REPRESENTATION --max_det $MAX_DET --confidence $CONF_THRESHOLD --iou $IOU_THRESHOLD

echo "INT8..."
mkdir -p $INT8_MODEL/trt-cache
TVM_TENSORRT_CACHE_DIR=$INT8_MODEL/trt-cache TVM_TENSORRT_USE_INT8=1 TRT_INT8_PATH=$INT8_ACTIVATIONS python3 infer.py --path_to_model $INT8_MODEL --input_image $IMAGE_PATH --labels $LABELS_PATH --format $FORMAT --representation $REPRESENTATION --max_det $MAX_DET --confidence $CONF_THRESHOLD --iou $IOU_THRESHOLD
