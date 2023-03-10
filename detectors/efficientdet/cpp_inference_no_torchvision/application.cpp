// ******************************************************************************
// Copyright (c) 2019-2023 by Latent AI Inc. All Rights Reserved.
//
// This file is part of the example-applications (LRE) product,
// and is released under the Apache 2.0 License.
// *****************************************************************************/

#include <tvm/runtime/latentai/lre_model.hpp>
#include <tvm/runtime/latentai/lre_cryption_service.hpp>

#include "efficientdet_processors.hpp"

#include <sys/time.h>
#include <experimental/filesystem>

namespace fs = std::experimental::filesystem;


int main(int argc, char *argv[]) {
  struct timeval t0, t1, t2, t3;
  
  // Parsing arguments by user
  std::string model_binary{argv[1]};
  std::string image_to_infer{argv[2]};
  std::vector <unsigned char> key;
  

  // Model Factory
  DLDevice device_t{kDLCUDA, 0}; //Change to kDLCPU if inference target is a CPU 
  LreModel model(model_binary,key, device_t);

 
  // Preprocessing
  gettimeofday(&t0, 0);
  std::cout << "Image: " << image_to_infer << std::endl;
  cv::Mat image_input = cv::imread(image_to_infer);
  cv::Mat processed_image =  preprocess_efficientdet(image_input,cv::Size (model.input_width,model.input_height));

  // Inference
  gettimeofday(&t1, 0);
  model.InferOnce(processed_image.data);
  gettimeofday(&t2, 0);

  // Post Processing
  auto result = postprocess_efficientdet(model.tvm_outputs,cv::Size (model.input_width,model.input_height));

  gettimeofday(&t3, 0);

  std::cout << " ----------------Boxes---------------------Scores----Label" << std::endl;
  std::cout << result[0] << std::endl;
  draw_boxes(result[0], image_to_infer,model.input_width, model.input_height);


  std::cout << std::setprecision(2) << std::fixed;
  std::cout << "Timing: " << (t1.tv_sec - t0.tv_sec) * 1000 + (t1.tv_usec - t0.tv_usec) / 1000.f << " ms pre process" << std::endl;
  std::cout << "Timing: " << (t2.tv_sec - t1.tv_sec) * 1000 + (t2.tv_usec - t1.tv_usec) / 1000.f << " ms infer + copy image" << std::endl;
  std::cout << "Timing: " << (t3.tv_sec - t2.tv_sec) * 1000 + (t3.tv_usec - t2.tv_usec) / 1000.f << " ms post process" << std::endl;
  
}