// ******************************************************************************
// Copyright (c) 2019-2023 by Latent AI Inc. All Rights Reserved.
//
// This file is part of the example-applications (LRE) product,
// and is released under the Apache 2.0 License.
// *****************************************************************************/

#include <tvm/runtime/latentai/lre.hpp>
#include <tvm/runtime/latentai/lre_cryption_service.hpp>

#include "efficientdet_processors.hpp"

#include <timer_chrono.hpp>
#include <image_manipulation.hpp>
#include <parse_inputs.hpp>
#include <display_model_metadata.hpp>
#include <iomanip>

Timer t_preprocessing,t_inference,t_op_transform,t_anchors_gen,t_decoding,t_nms;

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
  LRE::LatentRuntimeEngine model(model_binary);
  PrintModelMetadata(model);

  std::cout << "Image: " << imgPath << std::endl;

  // WarmUp Phase 
  model.warmUp(1);

  // Run pre, inference and post processing x iterations
  for (int j = 0; j < iterations; j++) {
    
    auto imageInput{ReadImage(imgPath)};
    cv::Scalar background(124, 116, 104);
    imageInput = ResizeImage(imageInput, model.input_width,model.input_height);

    // Preprocessing
    t_preprocessing.start();
    cv::Mat processed_image =  preprocess_efficientdet(imageInput);
    t_preprocessing.stop();

    // Infer
    t_inference.start();
    model.infer(processed_image.data);
    t_inference.stop();

    /*      Post Processing      */

    // Convert DLTensor to at::Tensor
    t_op_transform.start();
    auto outputs = convert_to_atTensor(model.getOutputs()[0]);
    t_op_transform.stop();

    auto tensors_ = effdet_tensors(outputs[0]);

    // NMS from Torchvision 
    auto result = batched_nms_coordinate_trick(tensors_["boxes"],tensors_["scores"],tensors_["classes"],0.45);

    // TopK Filter
    result = result.slice(0,0,100);

    tensors_["boxes"] = tensors_["boxes"].index({result});
    tensors_["classes"] = tensors_["classes"].index({result,at::indexing::None});
    tensors_["scores"] = tensors_["scores"].index({result,at::indexing::None});
    auto detections = at::cat({tensors_["boxes"],tensors_["scores"],tensors_["classes"]},1);

    // Confidence Threshold Filter 
    auto filtered_detections = at::where(detections.index({"...",4}) > 0.3);
    detections = detections.index({filtered_detections[0]});

    std::cout << "Detections:" << detections << std::endl;
    std::cout << "Width:" << model.input_width << std::endl;
    std::cout << "Height:" << model.input_height << std::endl;

    print_detections(detections);
    draw_boxes(detections, imgPath,model.input_width, model.input_height);
  }

  // std::cout << "Average Preprocessing Time: " << t_preprocessing.averageElapsedMilliseconds() << " ms" << std::endl;
  // std::cout << "Average Inference Time: " << t_inference.averageElapsedMilliseconds() << " ms" << std::endl;
  // std::cout << "Average Output Data Manipulation Time: " << t_op_transform.averageElapsedMilliseconds() << " ms" << std::endl;
  // std::cout << "Average Anchors Generation Time: " << t_anchors_gen.averageElapsedMilliseconds() << " ms" << std::endl;
  // std::cout << "Average Decoding Time: " << t_decoding.averageElapsedMilliseconds() << " ms" << std::endl;
  // std::cout << "Average NMS Time: " << t_nms.averageElapsedMilliseconds() << " ms" << std::endl;


}