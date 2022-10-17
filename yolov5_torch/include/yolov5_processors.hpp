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




static void monly_deleter(DLManagedTensor* self) { delete self; }

cv::Mat preprocess_yolov5(cv::Mat &ImageInput, float width, float height);

std::vector<at::Tensor> postprocess_yolov5(std::vector<DLTensor *> &tvm_outputs);
