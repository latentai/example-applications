// ******************************************************************************
// Copyright (c) 2019-2023 by Latent AI Inc. All Rights Reserved.
//
// This file is part of the example-applications (LRE) product,
// and is released under the Apache 2.0 License.
// *****************************************************************************/

#include <tvm/runtime/latentai/lre.hpp>
#include <tvm/runtime/latentai/lre_cryption_service.hpp>
#include "imagenet_torch_nchw_processors.hpp"

#include <timer_chrono.hpp>
#include <image_manipulation.hpp>
#include <parse_inputs.hpp>
#include <display_model_metadata.hpp>

Timer t_preprocessing,t_inference,t_postprocessing;

int main(int argc, char *argv[]) {
  InputParams params;
  if (!ParseInputs(argc, argv, InputType::Classifier, params)) {
      std::cerr << "Parsing of given command line arguments failed.\n";
      return 1;
  }

  std::string path_to_model = params.model_binary_path;
  int iterations = params.iterations;
  std::string img_path = params.input_image_path;
  std::string label_file_name = params.label_file_path;


  // Model Factory 
  LRE::LatentRuntimeEngine model_runtime(path_to_model);
  PrintModelMetadata(model_runtime);
  
  // WarmUp Phase 
  model_runtime.warmUp(1);

  std::pair<float, float> top_one;
  auto input_details = getLayoutDims(model_runtime.getInputLayouts(),model_runtime.getInputShapes());
  
  // Run pre, inference and post processing x iterations
  for (int i = 1; i < iterations; i++) {

    auto image_input{ReadImage(img_path)};
    auto resized_image = ResizeImage(image_input, input_details['W'], input_details['H']); //TODO: Parse from the string getInputShapes()  

    t_preprocessing.start();
    cv::Mat processed_image = preprocess_imagenet_torch_nchw(resized_image);
    t_postprocessing.stop();

    t_inference.start();
    model_runtime.infer(processed_image.data);
    t_inference.stop();

    auto output{model_runtime.getOutputs()};
    auto sizes{model_runtime.getOutputSizes()};
    t_postprocessing.start();
    top_one = postprocess_top_one(output, sizes);
    t_postprocessing.stop();

  }

  printTopOne(top_one, label_file_name);

  std::cout << "Average Preprocessing Time: " << t_preprocessing.averageElapsedMilliseconds() << " ms" << std::endl;
  std::cout << "Average Inference Time: " << t_inference.averageElapsedMilliseconds() << " ms" << std::endl;
  std::cout << "Average Postprocessing Time: " << t_postprocessing.averageElapsedMilliseconds() << " ms" << std::endl;

}
