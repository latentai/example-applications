// ******************************************************************************
// Copyright (c) 2019-2023 by Latent AI Inc. All Rights Reserved.
//
// This file is part of the example-applications (LRE) product,
// and is released under the Apache 2.0 License.
// *****************************************************************************/

#include "yolov5_processors.hpp"

cv::Mat preprocess_yolov5(cv::Mat &ImageInput, float width, float height) {
  cv::cvtColor(ImageInput, ImageInput, cv::COLOR_BGR2RGB);  // RGB Format required
  ImageInput.convertTo(ImageInput, CV_32FC3, 1.f / 255);  // Convert to float ranges 0-1
  cv::dnn::blobFromImage(ImageInput, ImageInput);  // NHWC to NCHW
  return ImageInput;
}

std::vector<at::Tensor> postprocess_yolov5(std::vector<DLTensor *> &tvm_outputs)
{
  std::vector<at::Tensor> dloutputs,result_output;

  // Decoding
  dloutputs = convert_to_atTensor(tvm_outputs);
  auto decoded_results = decode(dloutputs);
  auto scores = decoded_results[0];
  auto pred_boxes_x1y1x2y2 = decoded_results[1];

  // Drop Below Threshold
  auto inds_scores = at::where(scores > 0.45);
  pred_boxes_x1y1x2y2 = pred_boxes_x1y1x2y2.index({inds_scores[0]});
  scores = scores.index({inds_scores[0],inds_scores[1]});

  // NMS
  auto result = hard_nms(pred_boxes_x1y1x2y2, scores, inds_scores[1],0.45, -1, 200);
  auto filter = at::where(result.index({"...",4}) > 0.3);
  result_output.emplace_back(result.index({filter[0]}));
  return result_output;

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

void reshape_heads(std::vector<at::Tensor> &heads)
{
  for (int i = 0; i < 3 ; i++){

    auto dlTensorShape = heads[i].sizes();
    heads[i] = heads[i].reshape({dlTensorShape[0],-1,dlTensorShape[4]});
  }
}

std::vector<at::Tensor> decode(std::vector<at::Tensor> &heads)
{
  std::vector<at::Tensor> decoded;

  reshape_heads(heads);

  auto pred_logits =  at::sigmoid(at::cat({ heads[0],heads[1],heads[2]},1)[0]);
  auto scores = pred_logits.slice(1,5) * pred_logits.slice(1,4,5);
  
  auto pred_wh = (pred_logits.slice(1,0,2) * 2 + heads[3]) * heads[4];
  auto pred_xy = (pred_logits.slice(1,2,4) * 2).pow(2) * heads[5] * 0.5;

  auto pred_wh_unbinded = pred_wh.unbind(-1);
  auto pred_xy_unbinded = pred_xy.unbind(-1);

  auto x1 = pred_wh_unbinded[0] -  pred_xy_unbinded[0];
  auto y1 = pred_wh_unbinded[1] -  pred_xy_unbinded[1];
  auto x2 = pred_wh_unbinded[0] +  pred_xy_unbinded[0];
  auto y2 = pred_wh_unbinded[1] +  pred_xy_unbinded[1];

  auto pred_boxes_x1y1x2y2 = at::stack({x1,y1,x2,y2},-1);

  decoded.emplace_back(scores);
  decoded.emplace_back(pred_boxes_x1y1x2y2);

  return decoded;
}

at::Tensor hard_nms(at::Tensor select_boxes, at::Tensor scores, at::Tensor classes, float iou_threshold, int top_k, int candidates_size)
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

  return at::cat({select_boxes.index({picked_elements}),scores.index({picked_elements}).reshape({-1,1}), classes.index({picked_elements}).reshape({-1,1})},1);

}

at::Tensor get_area(at::Tensor left_top,at::Tensor right_bottom)
{
  auto hw = at::clamp(right_bottom - left_top,0.0f);
  return hw.index({"...",0}) * hw.index({"...",1});
}

at::Tensor get_iou(at::Tensor boxes0,at::Tensor boxes1,float eps)
{

  auto overlap_left_top{at::max(boxes0.slice(1,0,2),boxes1.slice(1,0,2))};
  auto overlap_right_bottom{at::min(boxes0.slice(1,2),boxes1.slice(1,2))};
  
  auto overlap_area = get_area(overlap_left_top,overlap_right_bottom);
  
  auto area0 = get_area(boxes0.slice(1,0,2),boxes1.slice(1,2));
  auto area1 = get_area(boxes1.slice(1,0,2),boxes1.slice(1,2));
  auto iou = overlap_area / (area0 + area1 - overlap_area + eps);
  return iou;

}

void draw_boxes(torch::Tensor pred_boxes_x1y1x2y2, std::string image_path, float width, float height)
{
  cv::Mat resized{};
  cv::Mat origImage = cv::imread(image_path);
  auto orig_size = cv::Size(origImage.rows, origImage.cols);

  // pre-processed image is too dark to see, we use original and re-size it
  cv::resize(origImage, resized, cv::Size(width, height)); 
  
  for (int i = 0; i < pred_boxes_x1y1x2y2.sizes()[0]; i++)
  {
    auto x1 = pred_boxes_x1y1x2y2[i][0].item<float>();
    auto y1 = pred_boxes_x1y1x2y2[i][1].item<float>();
    auto x2 = pred_boxes_x1y1x2y2[i][2].item<float>();
    auto y2 = pred_boxes_x1y1x2y2[i][3].item<float>();

    cv::rectangle(resized, cv::Point(x1, y1), cv::Point(x2, y2), cv::Scalar(255, 0, 0), 4, 8, 0);
  }
  image_path.replace(image_path.end()-4,image_path.end(), ("_" + date_stamp() + "_out.jpg"));
  cv::resize(resized, resized, cv::Size(origImage.cols, origImage.rows)); 
  std::cout << "Writing image to" << image_path << std::endl;
  cv::imwrite(image_path,resized);
}

void print_results(at::Tensor &result)
{
  std::cout << "-----------------------------------------------------------" << "\n";
  std::cout << std::right << std::setw(24) << "Box" 
            << std::right << std::setw(24) << "Score"
            << std::right << std::setw(12) << "Classes" << "\n";
  std::cout << "-----------------------------------------------------------" << "\n";
  std::cout << result;
  std::cout << "-----------------------------------------------------------" << "\n";

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