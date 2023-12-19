// ******************************************************************************
// Copyright (c) 2019-2023 by Latent AI Inc. All Rights Reserved.
//
// This file is part of the example-applications (LRE) product,
// and is released under the Apache 2.0 License.
// *****************************************************************************/

#include "processors.hpp"
#include "config.h"

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


at::Tensor clip_boxes_xyxy(at::Tensor boxes, at::Tensor size)
{
  boxes = boxes.clamp(0);
  auto sz = at::cat({size, size}, 0);
  boxes = boxes.min(sz);
  return boxes;
}

void draw_boxes(at::Tensor pred_boxes_x1y1x2y2, std::string image_path, float WIDTH, float HEIGHT)
{
  cv::Mat image_out{};
  cv::Mat origImage = cv::imread(image_path);
  cv::Scalar background(124, 116, 104);
  cv::Size dstSize(WIDTH,HEIGHT);
  #if YOLO
   image_out = resizeAndCenterImage(origImage, dstSize, background);
  #else
    cv::Size image_size = cv::Size(WIDTH, HEIGHT);
    cv::resize(origImage, image_out, image_size,0,0,cv::INTER_NEAREST); 
  #endif

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

std::vector<at::Tensor> convert_to_atTensors(std::vector<DLTensor *> dLTensors)
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


at::Tensor xywh2xyxy(const at::Tensor& x) {

    at::Tensor y = x.clone();

    // Compute top-left and bottom-right coordinates
    y.index({at::indexing::Slice(), 0}) = x.index({at::indexing::Slice(), 0}) - x.index({at::indexing::Slice(), 2}) / 2;  // top left x
    y.index({at::indexing::Slice(), 1}) = x.index({at::indexing::Slice(), 1}) - x.index({at::indexing::Slice(), 3}) / 2;  // top left y
    y.index({at::indexing::Slice(), 2}) = x.index({at::indexing::Slice(), 0}) + x.index({at::indexing::Slice(), 2}) / 2;  // bottom right x
    y.index({at::indexing::Slice(), 3}) = x.index({at::indexing::Slice(), 1}) + x.index({at::indexing::Slice(), 3}) / 2;  // bottom right y

    return y;
}

void clip_boxes(at::Tensor& boxes,  int img_height, int img_width) {
        boxes.index_put_({at::indexing::Slice(), 0}, at::clamp(boxes.index({at::indexing::Slice(), 0}), 0, img_width));  // x1
        boxes.index_put_({at::indexing::Slice(), 1}, at::clamp(boxes.index({at::indexing::Slice(), 1}), 0, img_height));  // y1
        boxes.index_put_({at::indexing::Slice(), 2}, at::clamp(boxes.index({at::indexing::Slice(), 2}), 0, img_width));  // x2
        boxes.index_put_({at::indexing::Slice(), 3}, at::clamp(boxes.index({at::indexing::Slice(), 3}), 0, img_height));  // y2
    }

at::Tensor decode_yolov5(const at::Tensor& preds, int img_height, int img_width) {
    at::Tensor bboxes, conf, scores;
    
    std::vector<at::Tensor> split_tensors = preds.split_with_sizes({4, 1, preds.size(2) - 5}, 2);
    
    // Access individual tensors
    bboxes = split_tensors[0];
    conf = split_tensors[1];
    scores = split_tensors[2];
    scores = scores * conf;  // final_conf = obj_conf * cls_conf

    // Apply coordinate transformations
    std::vector<at::Tensor> transformed_bboxes;
    for (int i = 0; i < bboxes.size(0); ++i) {
        at::Tensor bbox = bboxes.index({i, at::indexing::Slice(), at::indexing::Slice()});
        at::Tensor transformed_bbox = xywh2xyxy(bbox);
        clip_boxes(transformed_bbox, img_height, img_width);  // This is an inplace transformation
        transformed_bboxes.push_back(transformed_bbox);
    }

    at::Tensor transformed_bboxes_tensor = at::stack(transformed_bboxes, 0);

    // Concatenate the transformed bounding boxes and scores
    at::Tensor final_output = at::cat({transformed_bboxes_tensor, scores}, 2);

    return final_output;  // Format is batch_size x num_anchors x (4 + num_classes)
}

at::Tensor decode_yolov8(const at::Tensor& preds, int img_height, int img_width) {
    at::Tensor bboxes, scores;
    
    auto decoded_preds = preds.permute({0, 2, 1});

    std::vector<at::Tensor> split_tensors = decoded_preds.split_with_sizes({4, decoded_preds.size(2) - 4}, 2);
    
    // Access individual tensors
    bboxes = split_tensors[0];
    scores = split_tensors[1];

    // Apply coordinate transformations
    std::vector<at::Tensor> transformed_bboxes;
    for (int i = 0; i < bboxes.size(0); ++i) {
        at::Tensor bbox = bboxes.index({i, at::indexing::Slice(), at::indexing::Slice()});
        at::Tensor transformed_bbox = xywh2xyxy(bbox);
        clip_boxes(transformed_bbox, img_height, img_width);  // This is an inplace transformation
        transformed_bboxes.push_back(transformed_bbox);
    }

    at::Tensor transformed_bboxes_tensor = at::stack(transformed_bboxes, 0);

    // Concatenate the transformed bounding boxes and scores
    at::Tensor final_output = at::cat({transformed_bboxes_tensor, scores}, 2);

    return final_output;  // Format is batch_size x num_anchors x (4 + num_classes)
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
  effdet_tensors_["classes"] = output.index({at::indexing::Slice(),4});
  effdet_tensors_["scores"] = output.index({at::indexing::Slice(),5});

  return effdet_tensors_;
}

std::map<std::string, at::Tensor> yolo_tensors(at::Tensor output, float confidence_threshold)
{
  std::map<std::string, at::Tensor> yolo_tensors_;

  output = at::squeeze(output,0);

  yolo_tensors_["boxes"] = output.index({at::indexing::Slice(),at::indexing::Slice(0,4)});

  auto class_scores = output.index({at::indexing::Slice(),at::indexing::Slice(4,at::indexing::None)});

  yolo_tensors_["classes"] = std::get<1>(class_scores.max(1)).to(at::kFloat);
  yolo_tensors_["scores"] = std::get<0>(class_scores.max(1));

  auto filtered_scores = at::where(yolo_tensors_["scores"] > confidence_threshold);

  yolo_tensors_["boxes"] =   yolo_tensors_["boxes"].index({filtered_scores[0]});
  yolo_tensors_["classes"] =   yolo_tensors_["classes"].index({filtered_scores[0]});
  yolo_tensors_["scores"] =   yolo_tensors_["scores"].index({filtered_scores[0]});


  return yolo_tensors_;
}

std::map<std::string, at::Tensor> ssd_tensors(at::Tensor output, int width, int height)
{
  std::map<std::string, at::Tensor> ssd_tensors_;
  at::Tensor scale_tensor = at::empty(4);
  scale_tensor = scale_tensor.to(output.device());

  scale_tensor[0] = width; scale_tensor[2] = width;
  scale_tensor[1] = height; scale_tensor[3] = height;

  ssd_tensors_["boxes"] = output.index({at::indexing::Slice(),at::indexing::Slice(0,4)}) * scale_tensor;

  auto class_scores = output.index({at::indexing::Slice(),at::indexing::Slice(5,at::indexing::None)});

  ssd_tensors_["classes"] = std::get<1>(class_scores.max(1));
  ssd_tensors_["scores"] = std::get<0>(class_scores.max(1));


  return ssd_tensors_;
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
