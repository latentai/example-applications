/******************************************************************************
 * Copyright (c) 2019-2022 by Latent AI Inc. All Rights Reserved.
 *
 * This file is part of the latentai-lre (LRE) product,
 * and is released under the "Latent AI Commercial Software License".
 *****************************************************************************/

#include <tvm/runtime/latentai/lre_model.hpp>
#include "imagenet_torch_nchw_processors.hpp"


int main(int argc, char *argv[]) {
  // Parsing arguments by user

  std::string model_binary{argv[1]};
  std::string img_path{argv[2]};
  std::string label_file_name{argv[3]};

  // Model Factory
  DLDevice device_t{kDLCUDA, 0}; // If running in CPU change to kDLCPU
  LreModel model(model_binary, device_t);

  // Preprocessing
  cv::Mat image_input = cv::imread(img_path);
  cv::Mat processed_image = preprocess_imagenet_torch_nchw(image_input,model.input_width,model.input_height);

  // Inference
  model.InferOnce(processed_image.data);

  // Post Processing
  std::pair<float,float> top_val = postprocess_top_one(model.tvm_outputs,model.output_size,label_file_name);
}