// ******************************************************************************
// Copyright (c) 2019-2023 by Latent AI Inc. All Rights Reserved.
//
// This file is part of the example-applications (LRE) product,
// and is released under the Apache 2.0 License.
// *****************************************************************************/

#include <tvm/runtime/latentai/lre_model.hpp>
#include <tvm/runtime/latentai/lre_cryption_service.hpp>
#include "imagenet_torch_nchw_processors.hpp"

#include <timer_example.hpp>
#include <image_manipulation.hpp>
#include <parse_inputs.hpp>
#include <display_model_metadata.hpp>

TimeOperations classification_timer{};

TimeOperations RunClassification(const std::string& img_path, LreModel& model, std::string& label_file_name,
                  bool print_each_iteration, bool print_detections) {
    // Read and Resize Image
    auto image_input{ReadImage(img_path)};
    auto resized_image = ResizeImage(image_input, model.input_width, model.input_height);

    // Preprocessing
    classification_timer.preprocessing.emplace_back("Average Preprocessing", print_each_iteration);
    cv::Mat processed_image = preprocess_imagenet_torch_nchw(resized_image);
    classification_timer.preprocessing.back().Stop();

    // Inference
    classification_timer.inference.emplace_back("Average Inference", print_each_iteration);
    model.InferOnce(processed_image.data);
    classification_timer.inference.back().Stop();

    // Post Processing
    classification_timer.postprocessing.emplace_back("Average Postprocessing", print_each_iteration);
    std::pair<float, float> top_one = postprocess_top_one(model.tvm_outputs, model.output_size);
    classification_timer.postprocessing.back().Stop();

    if(print_detections){
      // Print Postprocessor output
      printTopOne(top_one, label_file_name);
    }
    return {classification_timer.preprocessing, classification_timer.inference, classification_timer.postprocessing};
}

int main(int argc, char *argv[]) {
  InputParams params;
  if (!ParseInputs(argc, argv, InputType::Classifier, params)) {
      std::cerr << "Parsing of given command line arguments failed.\n";
      return 1;
  }

  std::string model_binary = params.model_binary_path;
  int iterations = params.iterations;
  std::string img_path = params.img_path;
  std::string label_file_name = params.label_file_path;

  std::vector<unsigned char> key;
  bool print_each_iteration{true};
  bool dont_print_each_iteration{false};

  // Model Factory
  DLDevice device_t{kDLCUDA, 0}; // If running in CPU change to kDLCPU
  LreModel model(model_binary,key, device_t);
  PrintModelMetadata(model);
  
  // WarmUp Phase 
  Timer warmup_timer("Total Warm Up + Image Manipulation", print_each_iteration);
  RunClassification(img_path, model, label_file_name, dont_print_each_iteration, false);
  warmup_timer.Stop();
  
  // Run pre, inference and post processing x iterations
  for (int i = 1; i < iterations; i++) {
    int last_iteration{iterations - 1};
    bool print_detections{(i == last_iteration)}; // Print detections & stats only in last iteration
    classification_timer = RunClassification(img_path, model, label_file_name, dont_print_each_iteration, print_detections);
    if (print_detections){
      PrintOperationsStats(classification_timer, last_iteration);
    }
  }
}


  