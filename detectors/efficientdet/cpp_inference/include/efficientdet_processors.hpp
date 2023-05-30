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

namespace fs = std::experimental::filesystem;

static void monly_deleter(DLManagedTensor* self) { delete self; }

cv::Mat resizeAndCenterImage(const cv::Mat &input, const cv::Size &dstSize, const cv::Scalar &bgcolor);
std::vector<torch::Tensor> postprocess_efficientdet(std::vector<DLTensor *> &tvm_outputs,cv::Size dstSize);
cv::Mat preprocess_efficientdet(cv::Mat &imageInput);

torch::Tensor decode_box_outputs(torch::Tensor rel_codes, torch::Tensor anchors, bool output_xyxy);
torch::Tensor clip_boxes_xyxy(torch::Tensor boxes, torch::Tensor size);
torch::Tensor generate_anchors(int width, int height, int min_level, int max_level, int num_scales,c10::DeviceType infer_device);
void draw_boxes(torch::Tensor pred_boxes_x1y1x2y2, std::string image_path, float width, float height);
std::vector<at::Tensor> convert_to_atTensor(std::vector<DLTensor *> &dLTensors);
std::map<std::string, at::Tensor> get_top_classes_and_boxes(std::vector<torch::Tensor> outputs, int NUM_LEVELS = 5, int NUM_CLASSES = 90, int MAX_DETECTION_POINTS = 5000);
at::Tensor vision_nms(at::Tensor box_out_decoded,at::Tensor scores, at::Tensor classes, float iou_threshold = 0.45 , float confidence_threshold = 0.3 , int max_det_per_image = 15);
std::string date_stamp();

void print_detections(at::Tensor detections);