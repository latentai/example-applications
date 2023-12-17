// ******************************************************************************
// Copyright (c) 2019-2023 by Latent AI Inc. All Rights Reserved.
//
// This file is part of the example-applications (LRE) product,
// and is released under the Apache 2.0 License.
// *****************************************************************************/

#include <tvm/runtime/latentai/lre.hpp>
#include <tvm/runtime/latentai/lre_cryption_service.hpp>

#include "processors.hpp"

#include <timer_chrono.hpp>
#include <image_manipulation.hpp>
#include <parse_inputs.hpp>
#include <display_model_metadata.hpp>
#include <iomanip>

Timer t_preprocessing,t_inference,t_op_transform,t_nms, t_postprocessing;

int main(int argc, char *argv[]) {
  InputParams params;
  if (!ParseInputs(argc, argv, InputType::Detector, params)) {
      std::cerr << "Parsing of given command line arguments failed.\n";
      return 1;
  }

  std::string model_binary = params.model_binary_path;
  int iterations = params.iterations;
  std::string input_image_string = params.input_image_path;
  at::Tensor detections{};
  bool use_fp16 = std::getenv("TVM_TENSORRT_USE_FP16") ? std::stoi(std::getenv("TVM_TENSORRT_USE_FP16")) != 0 : false;
  
  // Model Factory
  LRE::LatentRuntimeEngine lre(model_binary);
  if (use_fp16){
    lre.setModelPrecision("float16");
    std::cout << "Running LRE as FP16" << std::endl;
  }
  PrintModelMetadata(lre);

  // WarmUp Phase 
  lre.warmUp(1);

  cv::Mat processed_image{};

  // Run pre, inference and post processing x iterations
  for (int j = 0; j < iterations; j++) {
    
    auto imageInput{ReadImage(input_image_string)};

    // Preprocessing
    t_preprocessing.start();
    if(MODEL=="NANODET" || MODEL=="EFFICIENTDET" || MODEL=="MOBNETSSD")
    {
     auto resized_image = ResizeImage(imageInput, lre.input_width, lre.input_height);
     processed_image = preprocess_efficientdet(resized_image);
    }
    else {
      if (MODEL=="YOLO")
      {
        cv::Scalar background(124, 116, 104);
        auto resized_and_centered_image = resizeAndCenterImage(imageInput, cv::Size(lre.input_width,lre.input_height), background);
        processed_image = preprocess_yolo(resized_and_centered_image);
      }
    }
  
    t_preprocessing.stop();

    // Infer
    t_inference.start();
    lre.infer(processed_image.data);
    t_inference.stop();

    // Post Processing
    t_postprocessing.start();
    t_op_transform.start();
    // Convert DLTensor to at::Tensor
    auto outputs = convert_to_atTensor(lre.getOutputs()[0]);
    std::map<std::string, at::Tensor> tensors_{};
    if (MODEL=="EFFICIENTDET"){
     tensors_ = effdet_tensors(outputs[0]);
    } 
    else{
      if (MODEL=="NANODET" || MODEL=="YOLO"){
        tensors_ = yolo_tensors(outputs[0]);
      }
      else{
      if (MODEL=="MOBNETSSD"){
        tensors_ = ssd_tensors(outputs[0], lre.input_width, lre.input_height);
        }
      }
    } 
    t_op_transform.stop();

    // NMS from Torchvision 
    t_nms.start();
    auto result = batched_nms_coordinate_trick(tensors_["boxes"],tensors_["scores"],tensors_["classes"], IOU_THRESHOLD);
    t_nms.stop();

    // TopK Filter
    result = result.slice(0,0,100);

    tensors_["boxes"] = tensors_["boxes"].index({result});
    tensors_["classes"] = tensors_["classes"].index({result,at::indexing::None});
    tensors_["scores"] = tensors_["scores"].index({result,at::indexing::None});
    detections = at::cat({tensors_["boxes"],tensors_["scores"],tensors_["classes"]},1);

    // Confidence Threshold Filter 
    auto filtered_detections = at::where(detections.index({"...",4}) > CONFIDENCE_THRESHOLD);
    detections = detections.index({filtered_detections[0]});
    t_postprocessing.stop();

  }
  print_detections(detections);
  draw_boxes(detections, input_image_string,lre.input_width, lre.input_height);

  std::cout << "Average Preprocessing Time: " << t_preprocessing.averageElapsedMilliseconds() << " ms" << std::endl;
  std::cout << "Average Inference Time: " << t_inference.averageElapsedMilliseconds() << " ms" << std::endl;
  std::cout << "Average Output Data Manipulation Time: " << t_op_transform.averageElapsedMilliseconds() << " ms" << std::endl;
  std::cout << "Average NMS Time: " << t_nms.averageElapsedMilliseconds() << " ms" << std::endl;
  std::cout << "Average Postprocessing Time: " << t_postprocessing.averageElapsedMilliseconds() << " ms" << std::endl;


}