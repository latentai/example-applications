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

  std::vector<at::Tensor> results;

  // Run pre, inference and post processing x iterations
  for (int j = 1; j < iterations; j++) {
    
    auto imageInput{ReadImage(imgPath)};
    cv::Scalar background(124, 116, 104);
    imageInput = resizeAndCenterImage(imageInput, cv::Size (model.input_width,model.input_height), background);

    // Preprocess
    t_preprocessing.start();
    cv::Mat processed_image =  preprocess_efficientdet(imageInput);
    t_preprocessing.stop();

    // Infer
    t_inference.start();
    model.infer(processed_image.data);
    t_inference.stop();

    // No SEGFAULT until here

    results.clear();

    t_op_transform.start();
    auto op = model.getOutputs();
    auto outputs = convert_to_atTensor(op);
    //auto clo_bx_in_cls = get_top_classes_and_boxes(outputs);
    // t_op_transform.stop();

    // t_anchors_gen.start();
    // torch::Tensor anchors = generate_anchors(model.input_height, model.input_width, 3, 7, 3,clo_bx_in_cls["box_outputs_all_after_topk"].device().type());
    // t_anchors_gen.stop();

    // int batch_size = outputs[0].sizes()[0];

  //   for (int i = 0; i < batch_size; i++)
  //   {
  //     t_decoding.start();
  //     auto class_out = clo_bx_in_cls["cls_outputs_all_after_topk"][i];
  //     auto box_out = clo_bx_in_cls["box_outputs_all_after_topk"][i];
  //     auto indices = clo_bx_in_cls["indices_all"][i];
  //     auto classes = clo_bx_in_cls["classes_all"][i];

  //     torch::Tensor anchor_boxes = anchors.index({indices, at::indexing::Slice(0, at::indexing::None)});
  //     auto box_out_decoded = decode_box_outputs(box_out, anchor_boxes, true);
  //     t_decoding.stop();
     
  //     t_nms.start();
  //     auto detections = hard_nms(box_out_decoded,class_out.sigmoid().squeeze(1),classes);
  //     auto filter = at::where(detections.index({"...",4}) > 0.3);
  //     detections = detections.index({filter[0]});
  //     t_nms.stop();
  //     results.emplace_back(detections);
  //   }
  }

  // print_detections(results[0]);
  // draw_boxes(results[0], imgPath,model.input_width, model.input_height);

  // std::cout << "Average Preprocessing Time: " << t_preprocessing.averageElapsedMilliseconds() << " ms" << std::endl;
  // std::cout << "Average Inference Time: " << t_inference.averageElapsedMilliseconds() << " ms" << std::endl;
  // std::cout << "Average Output Data Manipulation Time: " << t_op_transform.averageElapsedMilliseconds() << " ms" << std::endl;
  // std::cout << "Average Anchors Generation Time: " << t_anchors_gen.averageElapsedMilliseconds() << " ms" << std::endl;
  // std::cout << "Average Decoding Time: " << t_decoding.averageElapsedMilliseconds() << " ms" << std::endl;
  // std::cout << "Average NMS Time: " << t_nms.averageElapsedMilliseconds() << " ms" << std::endl;


}