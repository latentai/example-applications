# ******************************************************************************
# Copyright (c) 2019-2023 by Latent AI Inc. All Rights Reserved.
#
# This file is part of the example-applications (LRE) product,
# and is released under the Apache 2.0 License.
# *****************************************************************************/

#!/bin/bash

NANODET=0
MOBNETSSD=0
EFFICIENTDET=0
YOLO=1

FLOAT32_MODEL=~/models/detector/x86_64_cuda/Float32-compile
INT8_MODEL=~/models/detector/x86_64_cuda/Int8-optimize

if [ -v MODEL_PATH ];
then
    FLOAT32_MODEL=$MODEL_PATH/Float32-compile
    INT8_MODEL=$MODEL_PATH/Int8-optimize
fi

# Current known issue: These applications require a different libtorch
# (libtorch-cxx11-abi-shared-with-deps-2.0.1%2Bcu118.zip) installed that
# differs from the SDK version of libtorch.  The workaround is to install
# this version in the home directory for linking with the example
# applications when running in the SDK docker container.  This is the
# reason for the following workaround:

if [ -d ~/.torch-apps/libtorch ]
then
    TORCH_PATH=~/.torch-apps/libtorch
else
    TORCH_PATH=`python3 -c 'import torch;print(torch.utils.cmake_prefix_path)'`
fi
echo $TORCH_PATH

# Generate the config.h
echo "#ifndef CONFIG_H" > include/config.h
echo "#define CONFIG_H" >> include/config.h
echo "#define CONFIDENCE_THRESHOLD 0.30"  >> include/config.h # 0.3 normally, just for nanodet 0.45
echo "#define IOU_THRESHOLD 0.45"  >> include/config.h

echo "#define NANODET $NANODET" >> include/config.h
echo "#define MOBNETSSD $MOBNETSSD" >> include/config.h
echo "#define EFFICIENTDET $EFFICIENTDET" >> include/config.h
echo "#define YOLO $YOLO" >> include/config.h
echo "#endif // CONFIG_H" >> include/config.h

# Compile
mkdir build
cd build
cmake -DCMAKE_PREFIX_PATH=$TORCH_PATH ..
make -j 8
cd ..


# FP32
mkdir -p $FLOAT32_MODEL/trt-cache/
TVM_TENSORRT_CACHE_DIR=$FLOAT32_MODEL/trt-cache/ ./build/bin/application $FLOAT32_MODEL/modelLibrary.so 10 ../../sample_images/bus.jpg

# FP16
mkdir -p $FLOAT32_MODEL/trt-cache/
TVM_TENSORRT_CACHE_DIR=$FLOAT32_MODEL/trt-cache/ TVM_TENSORRT_USE_FP16=1 ./build/bin/application $FLOAT32_MODEL/modelLibrary.so 10 ../../sample_images/bus.jpg

# INT8
mkdir -p $INT8_MODEL/trt-cache/
TVM_TENSORRT_CACHE_DIR=$INT8_MODEL/trt-cache/ TVM_TENSORRT_USE_INT8=1 TRT_INT8_PATH=$INT8_MODEL/.activations/ ./build/bin/application $INT8_MODEL/modelLibrary.so 10 ../../sample_images/bus.jpg
