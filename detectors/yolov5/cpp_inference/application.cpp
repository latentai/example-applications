// *****************************************************************************
// Copyright (c) 2019-2023 by Latent AI Inc. All Rights Reserved.
//
// This file is part of the example-applications (LRE) product,
// and is released under the Apache 2.0 License.
// *****************************************************************************

#include <tvm/runtime/latentai/lre_model.hpp>
#include <tvm/runtime/latentai/lre_cryption_service.hpp>

#include "yolov5_processors.hpp"

#include <iomanip>
#include <timer_chrono.hpp>
#include <image_manipulation.hpp>
#include <parse_inputs.hpp>
#include <display_model_metadata.hpp>

Timer t_preprocessing,t_inference,t_decoding,t_thresholding,t_nms;
std::vector<at::Tensor> result_output(3), dloutputs;

int main(int argc, char *argv[]) {
  InputParams params;
  if (!ParseInputs(argc, argv, InputType::Detector, params)) {
      std::cerr << "Parsing of given command line arguments failed.\n";
      return 1;
  }
  std::string model_binary = params.model_binary_path;
  int iterations = params.iterations;
  std::string imgPath = params.img_path;

  
  // Model Factory
  DLDevice device_t{kDLCUDA, 0}; //Change to kDLCPU if inference target is a CPU 
  LreModel model(model_binary, device_t);
  PrintModelMetadata(model);

  std::cout << "Image: " << imgPath << std::endl;

  // WarmUp Phase 
  model.WarmUp(1);

  // Run pre, inference and post processing x iterations
  for (int i = 1; i < iterations; i++) {
    auto imageInput{ReadImage(imgPath)};
    cv::Scalar background(0, 0, 0);
    auto resized_image = resizeAndCenterImage(imageInput, cv::Size (model.input_width,model.input_height), background);

    // Preprocess
    t_preprocessing.start();
    cv::Mat processed_image =  preprocess_yolov5(resized_image,model.input_width,model.input_height);
    t_preprocessing.stop();

    // Infer
    t_inference.start();
    model.InferOnce(processed_image.data);
    t_inference.stop();

    // Postprocess
    /// decode
    t_decoding.start();
    dloutputs = convert_to_atTensor(model.tvm_outputs);
    auto decoded_results = decode(dloutputs);
    t_decoding.stop();

    /// drop below threshold
    t_thresholding.start();
    auto inds_scores = at::where(decoded_results[0] > 0.45);
    decoded_results[1] = decoded_results[1].index({inds_scores[0]}); //boxes
    decoded_results[0] = decoded_results[0].index({inds_scores[0],inds_scores[1]}); //scores
    t_thresholding.stop();

    /// NMS
    t_nms.start();
    auto result = vision::ops::nms(decoded_results[1],decoded_results[0],0.45);
    t_nms.stop();

    result_output[0] = decoded_results[1].index({result}); //boxes
    result_output[1] = decoded_results[0].index({result}); //scores
    result_output[2] = inds_scores[1].index({result});  //class

  }

  print_results(result_output);
  draw_boxes(result_output[0], imgPath,model.input_width, model.input_height);

  std::cout << "Average Preprocessing Time: " << t_preprocessing.averageElapsedMilliseconds() << " ms" << std::endl;
  std::cout << "Average Inference Time: " << t_inference.averageElapsedMilliseconds() << " ms" << std::endl;
  std::cout << "Average Decoding Time: " << t_decoding.averageElapsedMilliseconds() << " ms" << std::endl;
  std::cout << "Average Thresholding Time: " << t_thresholding.averageElapsedMilliseconds() << " ms" << std::endl;
  std::cout << "Average NMS Time: " << t_nms.averageElapsedMilliseconds() << " ms" << std::endl;

}