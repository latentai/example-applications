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

#include <torch/torch.h>
#include <torch/script.h>
#include <ATen/TensorIndexing.h>


#include <ATen/dlpack.h>
#include <ATen/DLConvertor.h>
#include <experimental/filesystem>

#include <ctime>

namespace fs = std::experimental::filesystem;

static void monly_deleter(DLManagedTensor* self) { delete self; }

cv::Mat resizeAndCenterImage(const cv::Mat &input, const cv::Size &dstSize, const cv::Scalar &bgcolor);
std::vector<torch::Tensor> postprocess_efficientdet(std::vector<DLTensor *> &tvm_outputs,cv::Size dstSize);
cv::Mat preprocess_efficientdet(cv::Mat &imageInput);

torch::Tensor decode_box_outputs(torch::Tensor rel_codes, torch::Tensor anchors, bool output_xyxy);
torch::Tensor clip_boxes_xyxy(torch::Tensor boxes, torch::Tensor size);
torch::Tensor generate_anchors(int width, int height, int min_level, int max_level, int num_scales,c10::DeviceType infer_device);
void draw_boxes(torch::Tensor pred_boxes_x1y1x2y2, std::string image_path, float width, float height);
at::Tensor convert_to_atTensor(DLTensor *tvm_output);

at::Tensor get_area(at::Tensor left_top,at::Tensor right_bottom);
at::Tensor get_iou(at::Tensor boxes0,at::Tensor boxes1,float eps = 1e-5);
at::Tensor hard_nms(at::Tensor select_boxes, at::Tensor scores, float iou_threshold, int top_k, int candidates_size, at::Tensor classes);

std::string date_stamp();