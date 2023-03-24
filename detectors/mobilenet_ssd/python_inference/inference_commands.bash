# ******************************************************************************
# Copyright (c) 2019-2023 by Latent AI Inc. All Rights Reserved.
#
# This file is part of the example-applications (LRE) product,
# and is released under the Apache 2.0 License.
# *****************************************************************************/

#!/bin/bash

FLOAT32_PACKAGE=~/models/mb1-ssd/x86_64_cuda/Float32-package
INT8_PACKAGE=~/models/mb1-ssd/x86_64_cuda/Int8-package

if [ -v MODEL_PATH ];
then
    FLOAT32_PACKAGE=$MODEL_PATH/Float32-package
    INT8_PACKAGE=$MODEL_PATH/Int8-package
fi

echo "FP32..."
mkdir -p $FLOAT32_PACKAGE/trt-cache
TVM_TENSORRT_CACHE_DIR=$FLOAT32_PACKAGE/trt-cache python3 infer.py --lre_object $FLOAT32_PACKAGE  --input_image ../../../sample_images/bus.jpg --labels ../../../labels/pascal_voc.txt

echo "FP16..."
mkdir -p $FLOAT32_PACKAGE/trt-cache
TVM_TENSORRT_CACHE_DIR=$FLOAT32_PACKAGE/trt-cache TVM_TENSORRT_USE_FP16=1 python3 infer.py --lre_object $FLOAT32_PACKAGE  --input_image ../../../sample_images/bus.jpg --labels ../../../labels/pascal_voc.txt

echo "INT8..."
mkdir -p $INT8_PACKAGE/trt-cache
TVM_TENSORRT_CACHE_DIR=$INT8_PACKAGE/trt-cache TVM_TENSORRT_USE_INT8=1 TRT_INT8_PATH=~/.latentai/LRE/.model/.activations/ python3 infer.py --lre_object $INT8_PACKAGE --input_image ../../../sample_images/bus.jpg --labels ../../../labels/pascal_voc.txt
