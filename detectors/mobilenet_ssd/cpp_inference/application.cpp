// ******************************************************************************
// Copyright (c) 2019-2023 by Latent AI Inc. All Rights Reserved.
//
// This file is part of the example-applications (LRE) product,
// and is released under the Apache 2.0 License.
// *****************************************************************************/

#include <tvm/runtime/latentai/lre_model.hpp>
#include <tvm/runtime/latentai/lre_cryption_service.hpp>

#include "mobilenetSSD_processors.hpp"
#include <timer_chrono.hpp>

#include <image_manipulation.hpp>
#include <parse_inputs.hpp>
#include <display_model_metadata.hpp>


Timer t_preprocessing,t_inference,t_op_transform,t_box_resize,t_thresholding,t_nms,t_filter;

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

  model.WarmUp(1);

  int label_nums;

  at::Tensor results;

  for (int j = 1; j < iterations; j++) {

    auto imageInput{ReadImage(imgPath)};
    auto resized_image = ResizeImage(imageInput, model.input_width, model.input_height);

    //Pre processing
    t_preprocessing.start();
    cv::Mat processed_image =  preprocess_mobilenetSSD(resized_image);
    t_preprocessing.stop();

    // Infer
    t_inference.start();
    model.InferOnce(processed_image.data);
    t_inference.stop();

    // Post Processing
    std::vector<at::Tensor> result_output,dloutputs;

    // Convert DLTensor to at::Tensor
    t_op_transform.start();
    dloutputs = convert_to_atTensor(model.tvm_outputs);
    // Get Raw Scores and Boxes
    auto scores = dloutputs[0][0];
    auto boxes = dloutputs[1][0];
    t_op_transform.stop();

    label_nums = scores.size(1);

    for(int i = 1; i < scores.size(1); i++)
    {
      // Drop classes and boxes below score threshold
      t_thresholding.start();
      auto probs = scores.index({"...", i});
      auto mask = at::where(probs > 0.1);

      probs = probs.index({mask[0]}); 
      if(probs.size(0) == 0){
        t_thresholding.stop();
        continue;
      }
      auto select_boxes = boxes.index({mask[0],"..."});
      t_thresholding.stop();

      // NMS
      t_nms.start();
      at::Tensor picked_boxes_and_scores = hard_nms(select_boxes,probs);
      t_nms.stop();

      // Box resize and label
      t_box_resize.start();
      resize_boxes_and_label(picked_boxes_and_scores,model.input_width,model.input_height,i);
      result_output.emplace_back(picked_boxes_and_scores);
      t_box_resize.stop();
    }

    // Result Filter
    t_filter.start();
    results = at::cat(result_output,0);
    auto filter = at::where(results.index({"...",4}) > 0.3);
    results = results.index({filter[0]});
    t_filter.stop();
  }

  print_results(results);
  draw_boxes(results, imgPath,model.input_width, model.input_height);

  std::cout << "Average Preprocessing Time: " << t_preprocessing.averageElapsedMilliseconds() << " ms" << std::endl;
  std::cout << "Average Inference Time: " << t_inference.averageElapsedMilliseconds() << " ms" << std::endl;
  std::cout << "Average Output Data Manipulation Time: " << t_op_transform.averageElapsedMilliseconds() << " ms" << std::endl;
  std::cout << "Average Thresholding Time: " << t_thresholding.averageElapsedMilliseconds() * label_nums << " ms" << std::endl;
  std::cout << "Average NMS Time: " << t_nms.averageElapsedMilliseconds() * label_nums << " ms" << std::endl;
  std::cout << "Average Box Resize and Label Time " << t_box_resize.averageElapsedMilliseconds() * label_nums << " ms" << std::endl;
  std::cout << "Average Filtering Time: " << t_filter.averageElapsedMilliseconds() << " ms" << std::endl;

}