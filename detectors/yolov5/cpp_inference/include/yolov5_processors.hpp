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

#include <ATen/ATen.h>

#include <ATen/dlpack.h>
#include <ATen/DLConvertor.h>

#include <torchvision/vision.h>
#include <torchvision/ops/nms.h>
#include <torch/torch.h>

#include <experimental/filesystem>

#include <ctime>

namespace fs = std::experimental::filesystem;

static void monly_deleter(DLManagedTensor* self) { delete self; }

cv::Mat preprocess_yolov5(cv::Mat &ImageInput, float width, float height);

std::vector<at::Tensor> postprocess_yolov5(std::vector<DLTensor *> &tvm_outputs);

std::vector<at::Tensor> convert_to_atTensor(std::vector<DLTensor *> &dLTensors);
void reshape_heads(std::vector<at::Tensor> &heads);
std::vector<at::Tensor> decode(std::vector<at::Tensor> &heads);
void draw_boxes(torch::Tensor pred_boxes_x1y1x2y2, std::string image_path, float width, float height);
void print_results(std::vector<at::Tensor> &result);

std::string date_stamp();

