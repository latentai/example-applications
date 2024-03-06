// ******************************************************************************
// Copyright (c) 2019-2023 by Latent AI Inc. All Rights Reserved.
//
// This file is part of the example-applications (LRE) product,
// and is released under the Apache 2.0 License.
// *****************************************************************************/

#include <tvm/runtime/latentai/lre.hpp>
#include <tvm/runtime/latentai/lre_cryption_service.hpp>

#include <nlohmann/json.hpp>

#include "processors.hpp"

#include <timer_chrono.hpp>
#include <parse_inputs.hpp>
#include <display_model_metadata.hpp>
#include <iomanip>

Timer t_preprocessing,t_inference,t_nms, t_postprocessing;


int main(int argc, char *argv[]) {
  
  std::string model_path, img_path, model_family,precision;
  int iterations;
  float conf_thres,iou_thres;

  auto cmdArgs = parseInput(argc, argv);

  // Extract and display required arguments
  iou_thres = std::stof(cmdArgs["--iou_thres"]);
  conf_thres = std::stof(cmdArgs["--conf_thres"]);
  model_path = cmdArgs["--model_path"];
  img_path = cmdArgs["--img_path"];
  iterations = std::stoi(cmdArgs["--iterations"]);
  model_family = cmdArgs["--model_family"];
  precision = cmdArgs["--precision"];


  at::Tensor detections{};  
  // Model Factory
  LRE::LatentRuntimeEngine lre(model_path);
  lre.setModelPrecision(precision);
  PrintModelMetadata(lre);

  // WarmUp Phase 
  lre.warmUp(100);

  cv::Mat processed_image{};

  // Run pre, inference and post processing x iterations
  for (int j = 0; j < iterations; j++) {
    
    auto imageInput = cv::imread(img_path);

    // Preprocessing
    if(model_family=="NANODET" || model_family=="MOBNETSSD")
    {
      cv::Mat resized_image;
      cv::resize(imageInput, resized_image, cv::Size(lre.input_width,lre.input_height),0,0,cv::INTER_NEAREST);

      t_preprocessing.start();
      processed_image = preprocess_efficientdet(resized_image);
    }
    else if(model_family=="EFFICIENTDET")
    {
      auto resized_and_centered_image = resizeAndCenterImage(imageInput, cv::Size(lre.input_width,lre.input_height), cv::Scalar(124, 116, 104));
      t_preprocessing.start();
      processed_image = preprocess_efficientdet(resized_and_centered_image);

    }
    else if (model_family=="YOLO")
    {
      auto resized_and_centered_image = resizeAndCenterImage(imageInput, cv::Size(lre.input_width,lre.input_height), cv::Scalar(124, 116, 104));

      t_preprocessing.start();
      processed_image = preprocess_yolo(resized_and_centered_image);
    }
    else
    {
        std::cerr << "Invalid model type, supported: YOLO, MOBNETSSD, EFFICIENTDET, NANODET\n";
    }
  
    t_preprocessing.stop();

    // Infer
    t_inference.start();
    lre.infer(processed_image.data);
    t_inference.stop();

    // Post Processing
    t_postprocessing.start();
    // Convert DLTensor to at::Tensor
    auto outputs = convert_to_atTensor(lre.getOutputs()[0]);
    std::map<std::string, at::Tensor> tensors_{};
    
    tensors_ = transform_tensors(outputs[0]);


    // Thresholding
    // Thresholding can have a huge impact on NMS timing. For improved Latency provide optimal thresholding score
    auto filtered_scores = at::where(tensors_["scores"] > conf_thres);
    
    if(filtered_scores[0].size(0) > 1){
      tensors_["boxes"] = tensors_["boxes"].index({filtered_scores[0]});
      tensors_["classes"] = tensors_["classes"].index({filtered_scores[0]});
      tensors_["scores"] = tensors_["scores"].index({filtered_scores[0]});
    }

    // NMS from Torchvision 
    t_nms.start();
    auto result = batched_nms_coordinate_trick(tensors_["boxes"],tensors_["scores"],tensors_["classes"], iou_thres);
    t_nms.stop();

    // TopK Filter
    result = result.slice(0,0,100);

    #ifndef HAVE_TORCHVISION
    if (result.defined() && result.scalar_type() != at::kLong) {
      result = result.to(at::kLong);
    }
    #endif

    tensors_["boxes"] = tensors_["boxes"].index({result});
    tensors_["classes"] = tensors_["classes"].index({result,at::indexing::None});
    tensors_["scores"] = tensors_["scores"].index({result,at::indexing::None});
    detections = at::cat({tensors_["boxes"],tensors_["scores"],tensors_["classes"]},1);

    t_postprocessing.stop();

  }

  // Print Detection and Draw Boxes Once
  print_detections(detections);
  draw_boxes(detections, img_path,lre.input_width, lre.input_height, model_family);

  // Get and Print Timings as JSON Object
  double avgPreprocessing = t_preprocessing.averageElapsedMilliseconds();
  double stdDevPreprocessing = t_preprocessing.standardDeviationMilliseconds();
  
  double avgInference = t_inference.averageElapsedMilliseconds();
  double stdDevInference = t_inference.standardDeviationMilliseconds();
  
  double avgNMS = t_nms.averageElapsedMilliseconds();
  double stdDevNMS = t_nms.standardDeviationMilliseconds();
  
  double avgPostprocessing = t_postprocessing.averageElapsedMilliseconds();
  double stdDevPostprocessing = t_postprocessing.standardDeviationMilliseconds();

  double sumOfAverageTimes = avgPreprocessing + avgInference + avgPostprocessing;

  // Create a JSON object
  json j = {
      {"UID", lre.getModelID()},
      {"Precision", lre.getModelPrecision()},
      {"Average Preprocessing Time ms", {{"Mean", roundToDecimalPlaces(avgPreprocessing,3)}, {"std_dev", roundToDecimalPlaces(stdDevPreprocessing,3)}}},
      {"Average Inference Time ms", {{"Mean", roundToDecimalPlaces(avgInference,3)}, {"std_dev", roundToDecimalPlaces(stdDevInference,3)}}},
      {"Average NMS Time ms", {{"Mean", roundToDecimalPlaces(avgNMS,3)}, {"std_dev", roundToDecimalPlaces(stdDevNMS,3)}}},
      {"Average Total Postprocessing Time ms", {{"Mean", roundToDecimalPlaces(avgPostprocessing,3)}, {"std_dev", roundToDecimalPlaces(stdDevPostprocessing,3)}}},
      {"Total Time ms", roundToDecimalPlaces(sumOfAverageTimes,3) }
  };
  
  // Print the JSON object
  std::cout << j.dump(2) << std::endl;

}
