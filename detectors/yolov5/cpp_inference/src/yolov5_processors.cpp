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

cv::Mat resizeAndCenterImage(const cv::Mat &input, const cv::Size &dstSize, const cv::Scalar &bgcolor)
{
  cv::Mat output;
  cv::Mat background(dstSize.width, dstSize.height, CV_32FC3, bgcolor);

  double h1 = dstSize.width * (input.rows / (double)input.cols);
  double w2 = dstSize.height * (input.cols / (double)input.rows);
  if (h1 <= dstSize.height)
  {
    cv::resize(input, output, cv::Size(dstSize.width, h1), cv::INTER_LINEAR);
  }
  else
  {
    cv::resize(input, output, cv::Size(w2, dstSize.height), cv::INTER_LINEAR);
  }

  double height, width ;

  if(output.cols < dstSize.width)
  {
    width = (dstSize.width - output.cols)/2;
  }
  else
  {
    width = 0;
  }

  if(output.rows < dstSize.height)
  {
    height = (dstSize.height - output.rows)/2;
  }
  else
  {
    height = 0;
  }

  cv::copyMakeBorder(output,output,height,height,width,width,cv::BORDER_CONSTANT,bgcolor);

  return output;
}

std::vector<at::Tensor> postprocess_yolov5(std::vector<DLTensor*> &tvm_outputs) {

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
  auto result = vision::ops::nms(pred_boxes_x1y1x2y2,scores,0.45);

  result_output.emplace_back(pred_boxes_x1y1x2y2.index({result}));
  result_output.emplace_back(scores.index({result}));
  result_output.emplace_back(inds_scores[1].index({result}));

  return result_output;
}



void draw_boxes(torch::Tensor pred_boxes_x1y1x2y2, std::string image_path, float width, float height)
{
  cv::Mat origImage = cv::imread(image_path);
  auto orig_size = cv::Size(origImage.rows, origImage.cols);

  cv::Scalar background(0, 0, 0);
  auto resized = resizeAndCenterImage(origImage, cv::Size (width,height), background);
  
  for (int i = 0; i < pred_boxes_x1y1x2y2.sizes()[0]; i++)
  {
    auto x1 = pred_boxes_x1y1x2y2[i][0].item<float>();
    auto y1 = pred_boxes_x1y1x2y2[i][1].item<float>();
    auto x2 = pred_boxes_x1y1x2y2[i][2].item<float>();
    auto y2 = pred_boxes_x1y1x2y2[i][3].item<float>();

    cv::rectangle(resized, cv::Point(x1, y1), cv::Point(x2, y2), cv::Scalar(255, 0, 0), 4, 8, 0);
  }
  image_path.replace(image_path.end()-4,image_path.end(), ("_" + date_stamp() + "_out.jpg"));
  std::cout << "Writing image to" << image_path << std::endl;
  cv::imwrite(image_path,resized);
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

std::string date_stamp()
{
  std::time_t curr_time;
	char date_string[100];
	
	std::time(&curr_time);
	std::tm * curr_tm{std::localtime(&curr_time)};
	std::strftime(date_string, 50, "%B_%d_%Y_%T", curr_tm);
  
  return date_string;
}

void print_results(std::vector<at::Tensor> &result){
  std::cout << "-----------------------------------------------------------" << "\n";
      std::cout << std::right << std::setw(24) << "Box" 
                << std::right << std::setw(24) << "Score"
                << std::right << std::setw(10) << "Class" << "\n";
      std::cout << "-----------------------------------------------------------" << "\n";
      for (int i = 0; i < result[0].size(0); i++) {
          std::cout << std::fixed << std::setprecision(4) 
                    << std::right << std::setw(8) << result[0][i][0].item<float>() << "  "
                    << std::right << std::setw(8) << result[0][i][1].item<float>() << "  "
                    << std::right << std::setw(8) << result[0][i][2].item<float>() << "  "
                    << std::right << std::setw(8) << result[0][i][3].item<float>() 
                    << std::right << std::setw(12) << std::setprecision(4) << result[1][i].item<float>()
                    << std::right << std::setw(8) << result[2][i].item<int>() << std::endl;
      }
}
