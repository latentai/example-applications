# ******************************************************************************
# Copyright (c) 2019-2023 by Latent AI Inc. All Rights Reserved.
#
# This file is part of the example-applications (LRE) product,
# and is released under the Apache 2.0 License.
# *****************************************************************************/

#!/bin/bash

FLOAT32_MODEL=/home/dev/models/recipe_hub/ssdn/aarch64_cuda_xavier_jp4/Float32-compile
# INT8_MODEL=$2
LABELS_PATH=/home/dev/src/example-applications/labels/pascal_voc.txt
MODEL_FORMAT=ssd
MAX_DET=5
CONF_THRESHOLD=0.3
IOU_THRESHOLD=0.3

IMAGE_PATH=../../sample_images/bus.jpg

if [ -v MODEL_PATH ];
then
    FLOAT32_MODEL=$MODEL_PATH/Float32-compile
    INT8_MODEL=$MODEL_PATH/Int8-optimize
fi
INT8_ACTIVATIONS=$INT8_MODEL/.activations/ # can we throw an error if this does not exist?

echo "FP32..."
mkdir -p $FLOAT32_MODEL/trt-cache
TVM_TENSORRT_CACHE_DIR=$FLOAT32_MODEL/trt-cache python3 infer.py --model_binary_path $FLOAT32_MODEL  --input_image_path $IMAGE_PATH  --labels $LABELS_PATH --model_format $MODEL_FORMAT --max_det $MAX_DET --confidence $CONF_THRESHOLD --iou $IOU_THRESHOLD

echo "FP16..."
mkdir -p $FLOAT32_MODEL/trt-cache
TVM_TENSORRT_CACHE_DIR=$FLOAT32_MODEL/trt-cache TVM_TENSORRT_USE_FP16=1 python3 infer.py --model_binary_path $FLOAT32_MODEL  --input_image_path $IMAGE_PATH --labels $LABELS_PATH --model_format $MODEL_FORMAT --max_det $MAX_DET --confidence $CONF_THRESHOLD --iou $IOU_THRESHOLD

echo "INT8..."
mkdir -p $INT8_MODEL/trt-cache
TVM_TENSORRT_CACHE_DIR=$INT8_MODEL/trt-cache TVM_TENSORRT_USE_INT8=1 TRT_INT8_PATH=$INT8_ACTIVATIONS python3 infer.py --model_binary_path $INT8_MODEL --input_image_path $IMAGE_PATH --labels $LABELS_PATH --model_format $MODEL_FORMAT --max_det $MAX_DET --confidence $CONF_THRESHOLD --iou $IOU_THRESHOLD
