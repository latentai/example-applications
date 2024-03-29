// ******************************************************************************
// Copyright (c) 2019-2023 by Latent AI Inc. All Rights Reserved.
//
// This file is part of the example-applications (LRE) product,
// and is released under the Apache 2.0 License.
// *****************************************************************************/

#include <dlpack/dlpack.h>
#include <tvm/runtime/module.h>
#include <tvm/runtime/packed_func.h>
#include <tvm/runtime/registry.h>

#include <fstream>

#include <opencv2/imgproc.hpp>
#include <opencv2/opencv.hpp>

#include <numeric>


cv::Mat preprocess_imagenet_torch_nchw(cv::Mat &t_image_input); 
std::vector<std::pair<float,float>> postprocess_top_five(std::vector<DLTensor *> &tvm_outputs, std::vector<int> &output_size);
std::pair<float,float> postprocess_top_one (std::vector<DLTensor *> &tvm_outputs, std::vector<int> &output_size);
void printTopOne(std::pair<float,float> top_one, std::string &label_file_name);
void printTopFive(std::vector<std::pair<float,float>> top_five, std::string &label_file_name);