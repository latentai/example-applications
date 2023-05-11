// ******************************************************************************
// Copyright (c) 2019-2023 by Latent AI Inc. All Rights Reserved.
//
// This file is part of the example-applications (LRE) product,
// and is released under the Apache 2.0 License.
// *****************************************************************************/

#include <tvm/runtime/latentai/lre_model.hpp>
#include <tvm/runtime/latentai/lre_cryption_service.hpp>

#include "mobilenetSSD_processors.hpp"
#include <timer_example.hpp>
#include <image_manipulation.hpp>
#include <parse_inputs.hpp>
#include <display_model_metadata.hpp>

TimeOperations mobnetSSDTimer{};

TimeOperations RunMobileNetSSDDetection(const std::string& imgPath, LreModel& model,
                  bool print_each_iteration, bool print_detections) {
  // Read Image
  auto imageInput{ReadImage(imgPath)};
  auto resized_image = ResizeImage(imageInput, model.input_width, model.input_height);

  // Preprocess
  mobnetSSDTimer.preprocessing.emplace_back("Average Preprocessing", print_each_iteration);
  cv::Mat processed_image =  preprocess_mobilenetSSD(resized_image, model.input_width, model.input_height);
  mobnetSSDTimer.preprocessing.back().Stop();

  // Infer
  mobnetSSDTimer.inference.emplace_back("Average Inference", print_each_iteration);
  model.InferOnce(processed_image.data);
  mobnetSSDTimer.inference.back().Stop();

  // Postprocess
  mobnetSSDTimer.postprocessing.emplace_back("Average Postprocessing", print_each_iteration);
  auto result = postprocess_mobilenetSSD(model.tvm_outputs);
  mobnetSSDTimer.postprocessing.back().Stop();
  
  if(print_detections){
    std::cout << "-----------------------------------------------------------" << "\n";
    std::cout << std::right << std::setw(24) << "Box" 
              << std::right << std::setw(24) << "Score"
              << std::right << std::setw(10) << "Class" << "\n";
    std::cout << "-----------------------------------------------------------" << "\n";
    std::cout << result << "\n";
    std::cout << "-----------------------------------------------------------" << "\n";
    draw_boxes(result, imgPath,model.input_width, model.input_height);
  }
  return {mobnetSSDTimer.preprocessing, mobnetSSDTimer.inference, mobnetSSDTimer.postprocessing};
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
  RunMobileNetSSDDetection(imgPath, model, dont_print_each_iteration, false);
  warmup_timer.Stop();

  // Run pre, inference and post processing x iterations
  for (int i = 1; i < iterations; i++) {
    int last_iteration{iterations - 1};
    bool print_detections{(i == last_iteration)}; // Print detections & stats only in last iteration
    mobnetSSDTimer = RunMobileNetSSDDetection(imgPath, model, dont_print_each_iteration, print_detections);
    if (print_detections){
      PrintOperationsStats(mobnetSSDTimer, last_iteration);
    }
  }
}