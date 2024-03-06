// ******************************************************************************
// Copyright (c) 2019-2023 by Latent AI Inc. All Rights Reserved.
//
// This file is part of the example-applications (LRE) product,
// and is released under the Apache 2.0 License.
// *****************************************************************************/

#include "processors.hpp"

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
  cv::subtract(imageInput, cv::Scalar(0.485f, 0.456f, 0.406f), imageInput, cv::noArray(), -1); //mean
  cv::divide(imageInput, cv::Scalar(0.229f, 0.224f, 0.225f), imageInput, 1, -1); //std
  cv::dnn::blobFromImage(imageInput, imageInput); // convert to nchw

  return imageInput;
}

cv::Mat preprocess_yolo(cv::Mat &ImageInput) {
  cv::cvtColor(ImageInput, ImageInput, cv::COLOR_BGR2RGB);  // RGB Format required
  ImageInput.convertTo(ImageInput, CV_32FC3, 1.f / 255);  // Convert to float ranges 0-1
  cv::dnn::blobFromImage(ImageInput, ImageInput);  // NHWC to NCHW
  return ImageInput;
}


torch::Tensor clip_boxes_xyxy(torch::Tensor boxes, torch::Tensor size)
{
  boxes = boxes.clamp(0);
  auto sz = torch::cat({size, size}, 0);
  boxes = boxes.min(sz);
  return boxes;
}

void draw_boxes(torch::Tensor pred_boxes_x1y1x2y2, std::string image_path, float WIDTH, float HEIGHT, std::string model_family)
{
  cv::Mat image_out{};
  cv::Mat origImage = cv::imread(image_path);
  cv::Scalar background(124, 116, 104);
  cv::Size dstSize(WIDTH,HEIGHT);
  if (model_family=="YOLO" || model_family == "EFFICIENTDET"){
    image_out = resizeAndCenterImage(origImage, dstSize, background);
  }
  else{
    cv::Size image_size = cv::Size(WIDTH, HEIGHT);
    cv::resize(origImage, image_out, image_size,0,0,cv::INTER_NEAREST); 
  }

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

at::Tensor convert_to_atTensor(DLTensor *dLTensor)
{

  DLManagedTensor* output = new DLManagedTensor{};
  output->dl_tensor = *dLTensor;
  output->deleter = &monly_deleter;

  auto op = at::fromDLPack(output);
  return op;
}

at::Tensor indicesToTensor(const std::vector<int>& indices, const at::TensorOptions& options) {
    at::Tensor result = at::empty({static_cast<int64_t>(indices.size())}, options);
    for (size_t i = 0; i < indices.size(); ++i) {
        result[i] = indices[i];
    }
    return result;
}

std::vector<float> tensorToScores(at::Tensor& scores) {
    std::vector<float> scores_vec(scores.size(0));
    std::memcpy(scores_vec.data(), scores.data_ptr<float>(), sizeof(float) * scores.size(0));
    return scores_vec;
}

std::vector<cv::Rect> tensorToRects(const at::Tensor& boxes) {
    std::vector<cv::Rect> rects;
    for (int i = 0; i < boxes.size(0); ++i) {
        int x1 = boxes[i][0].item<float>();
        int y1 = boxes[i][1].item<float>();
        int x2 = boxes[i][2].item<float>();
        int y2 = boxes[i][3].item<float>();
        rects.push_back(cv::Rect(x1, y1, x2 - x1, y2 - y1));
    }
    return rects;
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
  
  #ifdef HAVE_TORCHVISION
    auto result = vision::ops::nms(boxes_for_nms,scores,iou_threshold);
    return result;
  #endif

  #ifndef HAVE_TORCHVISION
    boxes = boxes.to(torch::kCPU);
    scores = scores.to(torch::kCPU);
    classes = classes.to(torch::kCPU);

    auto cv_boxes = tensorToRects(boxes_for_nms);
    auto cv_scores = tensorToScores(scores);
    std::vector<int> indices;
    cv::dnn::NMSBoxes(cv_boxes, cv_scores, 0.0, iou_threshold, indices);
    
    return indicesToTensor(indices, boxes.options());
  #endif


}


std::map<std::string, at::Tensor> transform_tensors(at::Tensor output)
{
  std::map<std::string, at::Tensor> transformed_tensors_;

  transformed_tensors_["boxes"] = output.index({at::indexing::Slice(),at::indexing::Slice(0,4)});

  auto class_scores = output.index({at::indexing::Slice(),at::indexing::Slice(4,at::indexing::None)});

  transformed_tensors_["classes"] = std::get<1>(class_scores.max(1)).to(torch::kFloat32);
  transformed_tensors_["scores"] = std::get<0>(class_scores.max(1));

  return transformed_tensors_;
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
