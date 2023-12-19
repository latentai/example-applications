// ******************************************************************************
// Copyright (c) 2019-2023 by Latent AI Inc. All Rights Reserved.
//
// This file is part of the example-applications (LRE) product,
// and is released under the Apache 2.0 License.
// *****************************************************************************/

#include <tvm/runtime/latentai/lre.hpp>
#include <tvm/runtime/latentai/lre_cryption_service.hpp>

#include "processors.hpp"
#include "config.h"

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
  std::string imgPath = params.img_path;
  at::Tensor detections{};
  
  // Model Factory
  LRE::LatentRuntimeEngine model(model_binary);
  PrintModelMetadata(model);

  // WarmUp Phase 
  model.warmUp(1);

  // Run pre, inference and post processing x iterations
  for (int j = 0; j < iterations; j++) {
    
    auto imageInput{ReadImage(imgPath)};

    // Preprocessing
    t_preprocessing.start();
    #if NANODET || EFFICIENTDET || MOBNETSSD
     auto resized_image = ResizeImage(imageInput, model.input_width, model.input_height);
     cv::Mat processed_image = preprocess_efficientdet(resized_image);
    #elif YOLO
     cv::Scalar background(124, 116, 104);
     auto resized_and_centered_image = resizeAndCenterImage(imageInput, cv::Size(model.input_width,model.input_height), background);
     cv::Mat processed_image = preprocess_yolo(resized_and_centered_image);
    #endif
    t_preprocessing.stop();

    // Infer
    t_inference.start();
    model.infer(processed_image.data);
    t_inference.stop();

    // Post Processing
    t_postprocessing.start();
    t_op_transform.start();


    // Convert DLTensor to at::Tensor
    auto outputs = convert_to_atTensor(model.getOutputs()[0]);
    #if EFFICIENTDET 
    auto tensors_ = effdet_tensors(outputs[0]);
    #elif NANODET || YOLO
    
    auto decoded_preds = decode_yolov8(outputs,model.input_width,model.input_height);
    
    auto tensors_ = yolo_tensors(decoded_preds,CONFIDENCE_THRESHOLD);
    #elif MOBNETSSD
    auto tensors_ = ssd_tensors(outputs[0], model.input_width, model.input_height);
    #endif
    t_op_transform.stop();

    // NMS from Torchvision 
    t_nms.start();
    auto result = batched_nms_coordinate_trick(tensors_["boxes"],tensors_["scores"],tensors_["classes"], 0.4);
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
  // draw_boxes(detections, imgPath,model.input_width, model.input_height);

  std::cout << "Average Preprocessing Time: " << t_preprocessing.averageElapsedMilliseconds() << " ms" << std::endl;
  std::cout << "Average Inference Time: " << t_inference.averageElapsedMilliseconds() << " ms" << std::endl;
  std::cout << "Average Output Data Manipulation Time: " << t_op_transform.averageElapsedMilliseconds() << " ms" << std::endl;
  std::cout << "Average NMS Time: " << t_nms.averageElapsedMilliseconds() << " ms" << std::endl;
  std::cout << "Average Postprocessing Time: " << t_postprocessing.averageElapsedMilliseconds() << " ms" << std::endl;


}