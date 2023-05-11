// ******************************************************************************
// Copyright (c) 2019-2023 by Latent AI Inc. All Rights Reserved.
//
// This file is part of the example-applications (LRE) product,
// and is released under the Apache 2.0 License.
// *****************************************************************************/

#include <tvm/runtime/latentai/lre_model.hpp>
#include <tvm/runtime/latentai/lre_cryption_service.hpp>

#include "yolov5_processors.hpp"

#include <iomanip>
#include <timer_example.hpp>
#include <image_manipulation.hpp>
#include <parse_inputs.hpp>
#include <display_model_metadata.hpp>

TimeOperations yoloTimer{};

TimeOperations RunYoloV5Detection(const std::string& imgPath, LreModel& model,
                  bool print_each_iteration, bool print_detections) {
  // Read Image
  auto imageInput{ReadImage(imgPath)};
  auto resized_image = ResizeImage(imageInput, model.input_width, model.input_height);

  // Preprocess
  yoloTimer.preprocessing.emplace_back("Average Preprocessing", print_each_iteration);
  cv::Mat processed_image =  preprocess_yolov5(resized_image,model.input_width,model.input_height);
  yoloTimer.preprocessing.back().Stop();

  // Infer
  yoloTimer.inference.emplace_back("Average Inference", print_each_iteration);
  model.InferOnce(processed_image.data);
  yoloTimer.inference.back().Stop();

  // Postprocess
  yoloTimer.postprocessing.emplace_back("Average Postprocessing", print_each_iteration);
  auto result = postprocess_yolov5(model.tvm_outputs,imgPath,model.input_width,model.input_height);
  yoloTimer.postprocessing.back().Stop();
  
  if(print_detections){
    std::cout << "-----------------------------------------------------------" << "\n";
    std::cout << std::right << std::setw(24) << "Box" 
              << std::right << std::setw(24) << "Score"
              << std::right << std::setw(10) << "Class" << "\n";
    std::cout << "-----------------------------------------------------------" << "\n";
    for (int i = 0; i < result[0].size(0); i++) {
        std::cout << std::fixed << std::setprecision(4) 
                  << std::right << std::setw(8) << result[0][i][0].item<float>() << "  "
                  << std::right << std::setw(8) << result[0][i][1].item<float>() << "  "
                  << std::right << std::setw(8) << result[0][i][2].item<float>() << "  "
                  << std::right << std::setw(8) << result[0][i][3].item<float>() 
                  << std::right << std::setw(12) << std::setprecision(4) << result[1][i].item<float>()
                  << std::right << std::setw(8) << result[2][i].item<int>() << std::endl;
    }
  draw_boxes(result[0], imgPath,model.input_width, model.input_height);
  }
  return {yoloTimer.preprocessing, yoloTimer.inference, yoloTimer.postprocessing};
}

int main(int argc, char *argv[]) {
  InputParams params;
  if (!ParseInputs(argc, argv, InputType::Detector, params)) {
      std::cerr << "Parsing of given command line arguments failed.\n";
      return 1;
  }

  std::string model_binary = params.model_binary_path;
  int iterations = params.iterations;
  std::string imgPath = params.img_path;

  // Uncomment next 3 lines to use LRE Cryption services to unlock the unecrypted key.
  
  // std::cout << " Enter password to unlock key " << std::endl;
  // std::cin >> password;
  // key = unlock_key(password,key_path);

  std::vector<unsigned char> key;
  bool print_each_iteration{true};
  bool dont_print_each_iteration{false};
  
  // Model Factory
  DLDevice device_t{kDLCUDA, 0}; //Change to kDLCPU if inference target is a CPU 
  LreModel model(model_binary,key, device_t);
  PrintModelMetadata(model);

  std::cout << "Image: " << imgPath << std::endl;

  // WarmUp Phase 
  Timer warmup_timer("Total Warm Up + Image Manipulation", print_each_iteration);
  RunYoloV5Detection(imgPath, model, dont_print_each_iteration, false);
  warmup_timer.Stop();

  // Run pre, inference and post processing x iterations
  for (int i = 1; i < iterations; i++) {
    int last_iteration{iterations - 1};
    bool print_detections{(i == last_iteration)}; // Print detections & stats only in last iteration
    yoloTimer = RunYoloV5Detection(imgPath, model, dont_print_each_iteration, print_detections);
    if (print_detections){
      PrintOperationsStats(yoloTimer, last_iteration);
    }
  }
}