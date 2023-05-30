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


#include <experimental/filesystem>

#include <ctime>

namespace fs = std::experimental::filesystem;

static void monly_deleter(DLManagedTensor* self) { delete self; }

cv::Mat preprocess_mobilenetSSD(cv::Mat &ImageInput);

at::Tensor postprocess_mobilenetSSD(std::vector<DLTensor *> &tvm_outputs);
void draw_boxes(at::Tensor pred_boxes_x1y1x2y2, std::string image_path,float width, float height);


std::vector<at::Tensor> convert_to_atTensor(std::vector<DLTensor *> &dLTensors);
void reshape_heads(std::vector<at::Tensor> &heads);
at::Tensor get_area(at::Tensor left_top,at::Tensor right_bottom);
at::Tensor get_iou(at::Tensor boxes0,at::Tensor boxes1,float eps = 1e-5);

void resize_boxes_and_label(at::Tensor &picked_boxes_and_scores, int width, int height, int label);
at::Tensor hard_nms(at::Tensor select_boxes, at::Tensor scores, float iou_threshold = 0.45, int top_k = -1, int candidates_size = 200);

std::string date_stamp();
void print_results(at::Tensor result);
