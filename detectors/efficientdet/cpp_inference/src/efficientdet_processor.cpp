// ******************************************************************************
// Copyright (c) 2019-2023 by Latent AI Inc. All Rights Reserved.
//
// This file is part of the example-applications (LRE) product,
// and is released under the Apache 2.0 License.
// *****************************************************************************/

#include "efficientdet_processors.hpp"


cv::Mat resizeAndCenterImage(const cv::Mat& input, const cv::Size& outputSize, const cv::Scalar& backgroundColor)
{
    CV_Assert(!input.empty() && input.type() == CV_8UC3);

    cv::Mat output;
    cv::Size2f inputSize = input.size();
    cv::Size2f outputSize2f = outputSize;
    cv::Size2f scaleFactor = cv::Size2f(outputSize2f.width / inputSize.width, outputSize2f.height / inputSize.height);
    float scale = std::min(scaleFactor.width, scaleFactor.height);

    cv::Size2f newSize = inputSize * scale;

    cv::resize(input, output, newSize, cv::INTER_LINEAR);
    cv::copyMakeBorder(output, output, (outputSize2f.height - newSize.height) / 2, (outputSize2f.height - newSize.height) / 2,
                       (outputSize2f.width - newSize.width) / 2, (outputSize2f.width - newSize.width) / 2, cv::BORDER_CONSTANT,
                       backgroundColor);

    return output;
}

cv::Mat preprocess_efficientdet(cv::Mat &imageInput)
{
  cv::cvtColor(imageInput, imageInput, cv::COLOR_BGR2RGB); // RGB Format required
  imageInput.convertTo(imageInput, CV_32FC3, 1.f/255.f); // Normalization between 0-1
  cv::subtract(imageInput, cv::Scalar(0.485f, 0.456f, 0.406f), imageInput, cv::noArray(), -1);
  cv::divide(imageInput, cv::Scalar(0.229f, 0.224f, 0.225f), imageInput, 1, -1);
  cv::dnn::blobFromImage(imageInput, imageInput); // convert to nchw

  return imageInput;
}

/*
torch::Tensor clip_boxes_xyxy(torch::Tensor boxes, torch::Tensor size)
{
  boxes = boxes.clamp(0);
  auto sz = torch::cat({size, size}, 0);
  boxes = boxes.min(sz);
  return boxes;
}

void draw_boxes(torch::Tensor pred_boxes_x1y1x2y2, std::string image_path, float WIDTH, float HEIGHT)
{
  cv::Mat origImage = cv::imread(image_path);
  cv::Scalar background(124, 116, 104);
  cv::Size dstSize(WIDTH,HEIGHT);
  cv::Mat image_out = resizeAndCenterImage(origImage, dstSize, background);

  for (int i = 0; i < pred_boxes_x1y1x2y2.sizes()[0]; i++)
  {
    auto x1 = pred_boxes_x1y1x2y2[i][0].item<float>();
    auto y1 = pred_boxes_x1y1x2y2[i][1].item<float>();
    auto x2 = pred_boxes_x1y1x2y2[i][2].item<float>();
    auto y2 = pred_boxes_x1y1x2y2[i][3].item<float>();

    cv::rectangle(image_out, cv::Point(x1, y1), cv::Point(x2, y2), cv::Scalar(255, 0, 0), 4, 8, 0);
  }
  image_path.replace(image_path.end()-4,image_path.end(), ("_" + date_stamp() + "_out.jpg"));
  std::cout << "Writing annotated image to " << image_path << std::endl;
  cv::imwrite(image_path,image_out);
}

std::vector<torch::Tensor> postprocess_efficientdet(std::vector<DLTensor *> &tvm_outputs,cv::Size dstSize)
{
  constexpr float PREDICTION_CONFIDENCE_THRESHOLD = 0.3;
  constexpr bool NMS_METHOD = false;
  constexpr int NUM_LEVELS = 5;
  constexpr int NUM_CLASSES = 90;
  constexpr int MAX_DETECTION_POINTS = 5000;
  constexpr int MAX_DET_PER_IMAGE = 15;

  std::vector<torch::Tensor> results;

  auto outputs = convert_to_atTensor(tvm_outputs);
  auto clo_bx_in_cls = get_top_classes_and_boxes(outputs);

  int batch_size = outputs[0].sizes()[0];

  for (int i = 0; i < batch_size; i++)
  {
    torch::Tensor detections;

    auto class_out = clo_bx_in_cls["cls_outputs_all_after_topk"][i];
    auto box_out = clo_bx_in_cls["box_outputs_all_after_topk"][i];
    auto indices = clo_bx_in_cls["indices_all"][i];
    auto classes = clo_bx_in_cls["classes_all"][i];

    torch::Tensor anchor_boxes = anchors.index({indices, at::indexing::Slice(0, at::indexing::None)});

    auto box_out_decoded = decode_box_outputs(box_out, anchor_boxes, true);

    auto scores = class_out.sigmoid().squeeze(1); 
    auto result = vision::ops::nms(box_out_decoded,scores,0.45);
    
    result = result.slice(0,0,MAX_DET_PER_IMAGE);
    
    box_out_decoded = box_out_decoded.index({result});
    classes = classes.index({result,torch::indexing::None});
    scores = scores.index({result,torch::indexing::None});
    detections = torch::cat({box_out_decoded,scores,classes},1);

    auto filtered_detections = torch::where(detections.index({"...",4}) > PREDICTION_CONFIDENCE_THRESHOLD);
    detections = detections.index({filtered_detections[0]});

    results.emplace_back(detections);
  }
  return results;
}
*/
at::Tensor convert_to_atTensor(DLTensor *dLTensor)
{

  DLManagedTensor* output = new DLManagedTensor{};
  output->dl_tensor = *dLTensor;
  output->deleter = &monly_deleter;

  auto op = at::fromDLPack(output);
  return op;
}

at::Tensor batched_nms_coordinate_trick(at::Tensor &boxes, at::Tensor &scores, at::Tensor &classes, float iou_threshold)
{
  if(boxes.numel() == 0)
  {
    return at::empty({0}, at::kFloat).to(boxes.device());
  }

  auto max_coordinate = boxes.max();
  auto offsets =  classes * (max_coordinate + at::ones({1}).to(boxes.device()));
  auto boxes_for_nms = boxes + offsets.index({at::indexing::Slice(), at::indexing::None});
  auto result = vision::ops::nms(boxes_for_nms,scores,iou_threshold);

  return result;

}

std::map<std::string, at::Tensor> effdet_tensors(at::Tensor output)
{
  std::map<std::string, at::Tensor> effdet_tensors_;

  effdet_tensors_["boxes"] = output.index({at::indexing::Slice(),at::indexing::Slice(0,4)});

  std::cout << effdet_tensors_["boxes"].sizes() << std::endl; 
  effdet_tensors_["classes"] = output.index({at::indexing::Slice(),4});
    std::cout << effdet_tensors_["classes"].sizes() << std::endl; 

  effdet_tensors_["scores"] = output.index({at::indexing::Slice(),5});

  return effdet_tensors_;
}

std::map<std::string, at::Tensor> yolo_tensors(at::Tensor output)
{
  std::map<std::string, at::Tensor> yolo_tensors_;

  yolo_tensors_["boxes"] = output.index({at::indexing::Slice(),at::indexing::Slice(0,4)});

    std::cout << yolo_tensors_["boxes"].sizes() << std::endl;


  auto class_scores = output.index({at::indexing::Slice(),at::indexing::Slice(4,at::indexing::None)});

  std::cout << class_scores.sizes() << std::endl;

  // Find the indices of the maximum values along dimension 2
  yolo_tensors_["classes"] = std::get<1>(class_scores.max(1)).to(torch::kFloat32);;

  std::cout << yolo_tensors_["classes"].sizes() << std::endl;


  std::cout << yolo_tensors_["classes"].sizes() << std::endl;


  // Find the maximum values along dimension 1
  yolo_tensors_["scores"] = std::get<0>(class_scores.max(1));

    std::cout << yolo_tensors_["scores"].sizes() << std::endl;


  return yolo_tensors_;
}


/*

std::string date_stamp()
{
  std::time_t curr_time;
	char date_string[100];
	
	std::time(&curr_time);
	std::tm * curr_tm{std::localtime(&curr_time)};
	std::strftime(date_string, 50, "%B_%d_%Y_%T", curr_tm);
  
  return date_string;
}


torch::Tensor vision_nms(at::Tensor box_out_decoded,at::Tensor scores, at::Tensor classes, float iou_threshold, float confidence_threshold, int max_det_per_image)
{
  auto result = vision::ops::nms(box_out_decoded,scores,0.45);
    
  result = result.slice(0,0,max_det_per_image);

  box_out_decoded = box_out_decoded.index({result});
  classes = classes.index({result,torch::indexing::None});
  scores = scores.index({result,torch::indexing::None});
  torch::Tensor detections = torch::cat({box_out_decoded,scores,classes},1);

  auto filtered_detections = torch::where(detections.index({"...",4}) > confidence_threshold);
  detections = detections.index({filtered_detections[0]});

  return detections;

}

void print_detections(at::Tensor detections)
{
  std::cout << "-----------------------------------------------------------" << "\n";
  std::cout << std::right << std::setw(24) << "Box" 
            << std::right << std::setw(24) << "Score"
            << std::right << std::setw(10) << "Class" << "\n";
  std::cout << "-----------------------------------------------------------" << "\n";
  std::cout << detections << "\n";
  std::cout << "-----------------------------------------------------------" << "\n";
}

*/