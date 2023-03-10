// ******************************************************************************
// Copyright (c) 2019-2023 by Latent AI Inc. All Rights Reserved.
//
// This file is part of the example-applications (LRE) product,
// and is released under the Apache 2.0 License.
// *****************************************************************************/

#include "mobilenetSSD_processors.hpp"

cv::Mat preprocess_mobilenetSSD(cv::Mat &ImageInput, float width, float height) {
    
    const cv::Size image_size = cv::Size( width, height );
    cv::cvtColor(ImageInput, ImageInput, cv::COLOR_BGR2RGB);
    cv::resize(ImageInput, ImageInput, image_size,0,0,cv::INTER_NEAREST); 
    ImageInput.convertTo(ImageInput, CV_32FC3, 1.f/255.f); // Normalization between 0-1
    cv::subtract(ImageInput,cv::Scalar(0.485f, 0.456f, 0.406f),ImageInput, cv::noArray(), -1);
    cv::divide(ImageInput,cv::Scalar(0.229f, 0.224f, 0.225f),ImageInput,1,-1);
    cv::dnn::blobFromImage( ImageInput, ImageInput ); // convert to nchw 

    return ImageInput;
}

at::Tensor postprocess_mobilenetSSD(std::vector<DLTensor*> &tvm_outputs) {

  float score_threshold = 0.01;
  float iou_threshold = 0.45;
  float threshold = 0.3;
  int candidates_size = 200;
  int top_k = -1;
  float eps = 1e-5;

  int width = 300;
  int height = 300;

  std::vector<at::Tensor> result_output,dloutputs;
  at::Tensor results;

  // Convert DLTensor to at::Tensor
  dloutputs = convert_to_atTensor(tvm_outputs);

  // Get Raw Scores and Boxes
  auto scores = dloutputs[0][0];
  auto boxes = dloutputs[1][0];

  // Iterate over each class
  at::Tensor probs;
  std::vector<at::Tensor> mask;
  at::Tensor select_boxes;
  std::tuple<at::Tensor, at::Tensor> sorted_scores;

  for(int i = 1; i < scores.size(1); i++)
  {

    // Drop classes and boxes below score threshold
    probs = scores.index({"...", i});
    mask = at::where(probs > score_threshold);

    probs = probs.index({mask[0]}); 

    if(probs.size(0) == 0){
      continue;
    }
    select_boxes = boxes.index({mask[0],"..."});

   
    at::Tensor picked_boxes_and_scores = hard_nms(select_boxes,probs,iou_threshold,top_k,candidates_size);

    picked_boxes_and_scores.index({"...",0}) = picked_boxes_and_scores.index({"...",0}) * width; 
    picked_boxes_and_scores.index({"...",1}) = picked_boxes_and_scores.index({"...",1}) * height; 
    picked_boxes_and_scores.index({"...",2}) = picked_boxes_and_scores.index({"...",2}) * width; 
    picked_boxes_and_scores.index({"...",3}) = picked_boxes_and_scores.index({"...",3}) * height;


    at::Tensor label = at::ones(picked_boxes_and_scores.size(0),picked_boxes_and_scores.device().type()) * i;
    picked_boxes_and_scores = at::cat({picked_boxes_and_scores,label.reshape({-1,1})},1);
    result_output.emplace_back(picked_boxes_and_scores);
  }

  results = at::cat(result_output,0);

  auto filter = at::where(results.index({"...",4}) > threshold);

  return results.index({filter[0]});

}

std::vector<at::Tensor> convert_to_atTensor(std::vector<DLTensor *> &dLTensors)
{
  std::vector<at::Tensor> atTensors;
  for (int i = 0; i < dLTensors.size() ; i++){

    DLManagedTensor* output = new DLManagedTensor{};
    output->dl_tensor = *dLTensors[i];
    output->deleter = &monly_deleter;

    auto op = at::fromDLPack(output);
    atTensors.emplace_back(op);
  }
  return atTensors;
}

at::Tensor get_area(at::Tensor left_top,at::Tensor right_bottom)
{
  auto hw = at::clamp(right_bottom - left_top,0.0f);
  return hw.index({"...",0}) * hw.index({"...",1});
}

at::Tensor get_iou(at::Tensor boxes0,at::Tensor boxes1,float eps)
{

  auto overlap_left_top = at::max(boxes0.slice(1,0,2),boxes1.slice(1,0,2));
  auto overlap_right_bottom = at::min(boxes0.slice(1,2),boxes1.slice(1,2));

  auto overlap_area = get_area(overlap_left_top,overlap_right_bottom);

  auto area0 = get_area(boxes0.slice(1,0,2),boxes1.slice(1,2));
  auto area1 = get_area(boxes1.slice(1,0,2),boxes1.slice(1,2));

  auto iou = overlap_area / (area0 + area1 - overlap_area + eps);

  return iou;

}


at::Tensor hard_nms(at::Tensor select_boxes, at::Tensor scores, float iou_threshold, int top_k, int candidates_size)
{

  std::tuple<at::Tensor, at::Tensor> sorted_scores;
  std::vector <at::Tensor> picked;
  at::Tensor picked_elements;
  at::Tensor current;
  at::Tensor current_box;

  sorted_scores = scores.sort(-1,true);

  // Keep only top candidates_sizes scores
  std::get<0>(sorted_scores) = std::get<0>(sorted_scores).slice(0,0,candidates_size);
  std::get<1>(sorted_scores) = std::get<1>(sorted_scores).slice(0,0,candidates_size);

  while(std::get<1>(sorted_scores).size(0) > 0){

  current = std::get<1>(sorted_scores)[0];
  picked.emplace_back(current);

  if(0 < (top_k == picked.size()) or (std::get<1>(sorted_scores).size(0) == 1))
  {
    break;
  }

  current_box = select_boxes[current];

  std::get<0>(sorted_scores) = std::get<0>(sorted_scores).slice(0,1);
  std::get<1>(sorted_scores) = std::get<1>(sorted_scores).slice(0,1);

  auto rest_boxes = select_boxes.index({std::get<1>(sorted_scores)});
                                                                        
  // IOU
  auto iou = get_iou(rest_boxes,current_box.unsqueeze(0));

  // Drop indexes below threshold
  auto indexes = at::where(iou <= iou_threshold);
  
  std::get<0>(sorted_scores) = std::get<0>(sorted_scores).index({indexes[0]});
  std::get<1>(sorted_scores) = std::get<1>(sorted_scores).index({indexes[0]});
}

  picked_elements = at::stack(picked);

  return at::cat({select_boxes.index({picked_elements}),scores.index({picked_elements}).reshape({-1,1})},1);

}


void draw_boxes(at::Tensor pred_boxes_x1y1x2y2, std::string image_path,float width, float height)
{
  cv::Mat resized{};
  cv::Mat origImage = cv::imread(image_path);
  auto orig_size = cv::Size(origImage.rows, origImage.cols);
  cv::resize(origImage, resized, cv::Size(width, height),0,0,cv::INTER_NEAREST); 

  for (int i = 0; i < pred_boxes_x1y1x2y2.sizes()[0]; i++)
  {
    auto x1 = pred_boxes_x1y1x2y2[i][0].item<float>();
    auto y1 = pred_boxes_x1y1x2y2[i][1].item<float>();
    auto x2 = pred_boxes_x1y1x2y2[i][2].item<float>();
    auto y2 = pred_boxes_x1y1x2y2[i][3].item<float>();

    cv::rectangle(resized, cv::Point(x1, y1), cv::Point(x2, y2), cv::Scalar(255, 0, 0), 4, 8, 0);
  }
  image_path.replace(image_path.end()-4,image_path.end(), ("_" + date_stamp() + "_SSD_out.jpg"));
  cv::resize(resized, resized, cv::Size(origImage.cols, origImage.rows)); 
  std::cout << "Writing annotated image to " << image_path << std::endl;
  cv::imwrite(image_path,resized);
}

std::string date_stamp()
{
  std::time_t curr_time;
	char date_string[100];
	
	std::time(&curr_time);
	std::tm * curr_tm{std::localtime(&curr_time)};
	std::strftime(date_string, 50, "%B_%d_%Y_%T", curr_tm);
  
  return date_string;
}




