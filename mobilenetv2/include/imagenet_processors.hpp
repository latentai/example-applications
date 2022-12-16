/******************************************************************************
 * Copyright (c) 2019-2022 by Latent AI Inc. All Rights Reserved.
 *
 * This file is part of the latentai-lre (LRE) product,
 * and is released under the "Latent AI Commercial Software License".
 *****************************************************************************/

#include <dlpack/dlpack.h>
#include <tvm/runtime/module.h>
#include <tvm/runtime/packed_func.h>
#include <tvm/runtime/registry.h>

#include <fstream>

#include <opencv2/imgproc.hpp>
#include <opencv2/opencv.hpp>

#include <numeric>


cv::Mat preprocess_imagenet(cv::Mat &t_image_input, float image_shape_height,float image_shape_width); 
std::vector<std::pair<float,float>> postprocess_top_five(std::vector<DLTensor *> &tvm_outputs, std::vector<int> &output_size, std::string &label_file_name);
std::pair<float,float> postprocess_top_one (std::vector<DLTensor *> &tvm_outputs, std::vector<int> &output_size, std::string &label_file_name);
