# ******************************************************************************
# Copyright (c) 2019-2023 by Latent AI Inc. All Rights Reserved.
#
# This file is part of the example-applications (LRE) product,
# and is released under the Apache 2.0 License.
# *****************************************************************************/

#!/bin/bash

FLOAT32_MODEL=~/models/timm-gernet_m/x86_64_cuda/Float32-compile
INT8_MODEL=~/models/timm-gernet_m/x86_64_cuda/Int8-optimize

if [ -v MODEL_PATH ];
then
    FLOAT32_PACKAGE=$MODEL_PATH/Float32-package
    INT8_PACKAGE=$MODEL_PATH/Int8-package
fi

echo "FP32..."
mkdir -p $FLOAT32_PACKAGE/trt-cache
TVM_TENSORRT_CACHE_DIR=$FLOAT32_PACKAGE/trt-cache python3 infer.py --path_to_model $FLOAT32_MODEL --input_image ../../sample_images/apple.jpg --labels ../../labels/class_names_10.txt

echo "FP16..."
mkdir -p $FLOAT32_PACKAGE/trt-cache
TVM_TENSORRT_CACHE_DIR=$FLOAT32_PACKAGE/trt-cache TVM_TENSORRT_USE_FP16=1 python3 infer.py --path_to_model $FLOAT32_MODEL --input_image ../../sample_images/apple.jpg --labels ../../labels/class_names_10.txt

echo "INT8..."
mkdir -p $INT8_PACKAGE/trt-cache
TVM_TENSORRT_CACHE_DIR=$INT8_PACKAGE/trt-cache TVM_TENSORRT_USE_INT8=1 TRT_INT8_PATH=~/.latentai/LRE/.model/.activations/ python3 infer.py --path_to_model $INT8_MODEL --input_image ../../sample_images/apple.jpg --labels ../../labels/class_names_10.txt
