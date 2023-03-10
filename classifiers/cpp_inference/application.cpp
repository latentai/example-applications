// ******************************************************************************
// Copyright (c) 2019-2023 by Latent AI Inc. All Rights Reserved.
//
// This file is part of the example-applications (LRE) product,
// and is released under the Apache 2.0 License.
// *****************************************************************************/

#include <tvm/runtime/latentai/lre_model.hpp>
#include <tvm/runtime/latentai/lre_cryption_service.hpp>
#include "imagenet_torch_nchw_processors.hpp"
#include <sys/time.h>


int main(int argc, char *argv[]) {
  struct timeval t0, t1, t2, t3;
  // Parsing arguments by user

  std::string model_binary{argv[1]};
  std::string img_path{argv[2]};
  std::string label_file_name{argv[3]};
  std::vector<unsigned char> key;

  // Model Factory
  DLDevice device_t{kDLCUDA, 0}; // If running in CPU change to kDLCPU
  LreModel model(model_binary,key, device_t);

  // Preprocessing
  gettimeofday(&t0, 0);
  cv::Mat image_input = cv::imread(img_path);
  cv::Mat processed_image = preprocess_imagenet_torch_nchw(image_input,model.input_width,model.input_height);

  // Inference
  gettimeofday(&t1, 0);
  model.InferOnce(processed_image.data);
  gettimeofday(&t2, 0);

  // Post Processing
  std::pair<float,float> top_val = postprocess_top_one(model.tvm_outputs,model.output_size,label_file_name);
  gettimeofday(&t3, 0);

  std::cout << std::setprecision(2) << std::fixed;
  std::cout << "Timing: " << (t1.tv_sec - t0.tv_sec) * 1000 + (t1.tv_usec - t0.tv_usec) / 1000.f << " ms pre process" << std::endl;
  std::cout << "Timing: " << (t2.tv_sec - t1.tv_sec) * 1000 + (t2.tv_usec - t1.tv_usec) / 1000.f << " ms infer + copy image" << std::endl;
  std::cout << "Timing: " << (t3.tv_sec - t2.tv_sec) * 1000 + (t3.tv_usec - t2.tv_usec) / 1000.f << " ms post process" << std::endl;

}