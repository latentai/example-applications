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

#include <torchvision/vision.h>
#include <torchvision/ops/nms.h>

#include <experimental/filesystem>

#include <ctime>

#define CONFIDENCE_THRESHOLD 0.30 // 0.3 normally, just for nanodet 0.45
#define IOU_THRESHOLD 0.45

// Initialize model types
#define NANODET 0
#define EFFICIENTDET 0
#define MOBNETSSD 0
#define YOLO 0

namespace fs = std::experimental::filesystem;

static void monly_deleter(DLManagedTensor* self) { delete self; }

cv::Mat resizeAndCenterImage(const cv::Mat &input, const cv::Size &dstSize, const cv::Scalar &bgcolor);
cv::Mat preprocess_efficientdet(cv::Mat &imageInput);
cv::Mat preprocess_yolo(cv::Mat &ImageInput);
 
at::Tensor convert_to_atTensor(DLTensor* dLTensor);
at::Tensor batched_nms_coordinate_trick(at::Tensor &boxes, at::Tensor &scores, at::Tensor &classes, float iou_threshold);

std::map<std::string, at::Tensor> effdet_tensors(at::Tensor output);
std::map<std::string, at::Tensor> yolo_tensors(at::Tensor output);
std::map<std::string, at::Tensor> ssd_tensors(at::Tensor output, int width, int height);


void draw_boxes(torch::Tensor pred_boxes_x1y1x2y2, std::string image_path, float width, float height);
std::string date_stamp();
void print_detections(at::Tensor detections);