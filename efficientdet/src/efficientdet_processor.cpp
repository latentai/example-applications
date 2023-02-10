#include "efficientdet_processors.hpp"


cv::Mat resizeKeepAspectRatio(const cv::Mat &input, const cv::Size &dstSize, const cv::Scalar &bgcolor)
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

cv::Mat preprocess_efficientdet(cv::Mat &imageInput, cv::Size dstSize)
{

  cv::cvtColor(imageInput, imageInput, cv::COLOR_BGR2RGB); // RGB Format required
  cv::Scalar background(124, 116, 104);
  imageInput = resizeKeepAspectRatio(imageInput, dstSize, background);
  imageInput.convertTo(imageInput, CV_32FC3, 1.f/255.f); // Normalization between 0-1
  cv::subtract(imageInput, cv::Scalar(0.485f, 0.456f, 0.406f), imageInput, cv::noArray(), -1);
  cv::divide(imageInput, cv::Scalar(0.229f, 0.224f, 0.225f), imageInput, 1, -1);
  cv::dnn::blobFromImage(imageInput, imageInput); // convert to nchw

  return imageInput;
}

torch::Tensor decode_box_outputs(torch::Tensor rel_codes, torch::Tensor anchors, bool output_xyxy)
{
  torch::Tensor out;
  torch::Tensor ty, tx, th, tw;


  auto ycenter_a = (anchors.index({"...",0}) + anchors.index({"...",2})) / 2;
  auto xcenter_a = (anchors.index({"...",1}) + anchors.index({"...",3})) / 2;

  auto ha = (anchors.index({"...",2}) - anchors.index({"...",0}));
  auto wa = (anchors.index({"...",3}) - anchors.index({"...",1}));

  std::vector<torch::Tensor> t = rel_codes.unbind(1);
  ty = t[0];
  tx = t[1];
  th = t[2];
  tw = t[3];
  // dimensions!!!
  auto w = torch::exp(tw) * wa;
  auto h = torch::exp(th) * ha;
  auto ycenter = ty * ha + ycenter_a;
  auto xcenter = tx * wa + xcenter_a;

  auto ymin = ycenter - h / 2.0;
  auto xmin = xcenter - w / 2.0;
  auto ymax = ycenter + h / 2.0;
  auto xmax = xcenter + w / 2.0;
  
  out = torch::stack({xmin, ymin, xmax, ymax}, 1);

  return out;
}

torch::Tensor clip_boxes_xyxy(torch::Tensor boxes, torch::Tensor size)
{
  boxes = boxes.clamp(0);
  auto sz = torch::cat({size, size}, 0);
  boxes = boxes.min(sz);
  return boxes;
}

torch::Tensor generate_anchors(int width, int height, int min_level, int max_level, int num_scales,c10::DeviceType infer_device)
{

  float anchor_scale = 4.0;
  std::vector<std::pair<float, float>> aspect_ratios = {{1, 1}, {1.4, 0.7}, {0.7, 1.4}};
  std::pair<int, int> image_size = {height, width};
  std::pair<int, int> feat_size = image_size;
  torch::Tensor anchor_scales = torch::ones(max_level - min_level + 1) * anchor_scale;

  // Get Feat Sizes *get_feat_sizes(image_size, max_level)
  std::vector<std::pair<int, int>> feat_sizes;
  feat_sizes.push_back(feat_size);
  for (int i = 1; i < (max_level + 1); i++)
  {
    feat_size.first = floor((feat_size.first - 1) / 2) + 1;
    feat_size.second = floor((feat_size.second - 1) / 2) + 1;
    feat_sizes.push_back(feat_size);
  }

  // See _generate_configs() in anchors.py
  std::map<int, std::vector<std::vector<float>>> anchor_configs;

  for (int level = min_level; level < max_level + 1; level++)
  {
    std::vector<std::vector<float>> lconf;
    for (int scale_octave = 0; scale_octave < num_scales; scale_octave++)
    {
      for (auto aspect : aspect_ratios)
      {
        std::vector<float> conf;
        conf.push_back(feat_sizes[0].first * 1.f / feat_sizes[level].first);
        conf.push_back(feat_sizes[0].second * 1.f / feat_sizes[level].first);
        conf.push_back((scale_octave * 1.f) / num_scales);
        conf.push_back(aspect.first);
        conf.push_back(aspect.second);
        conf.push_back(anchor_scales[level - min_level].item().toFloat() * 1.f);
        lconf.push_back(conf);
        conf.clear();
      }
    }
    anchor_configs[level] = lconf;
    lconf.clear();
  }

  // See _generate_boxes() in anchors.py
  std::vector<torch::Tensor> boxes_all;
  for (auto configs : anchor_configs)
  {
    std::vector<torch::Tensor> boxes_level;
    for (auto item : configs.second)
    {
      // std::cout << item[0] << "," << item[1] << "," << item[2] << "," << item[3] << ","<< item[4] << "," << item[5] << std::endl;
      float stride[] = {item[0], item[1]};
      float octave_scale = item[2];
      float aspect_x = item[3];
      float aspect_y = item[4];
      float anchor_scale = item[5];

      float base_anchor_size_x = anchor_scale * stride[1] * pow(2, octave_scale);
      float base_anchor_size_y = anchor_scale * stride[0] * pow(2, octave_scale);
      
      float anchor_size_x_2 = base_anchor_size_x * aspect_x / 2.0;
      float anchor_size_y_2 = base_anchor_size_y * aspect_y / 2.0;

      auto _x = torch::arange(stride[1] / 2, image_size.second, stride[1], infer_device);
      auto _y = torch::arange(stride[0] / 2, image_size.first, stride[0], infer_device);

      std::vector<torch::Tensor> xyv;
      xyv = torch::meshgrid({_x, _y}, "xy");

      xyv[0] = xyv[0].reshape(-1);
      xyv[1] = xyv[1].reshape(-1);

      torch::Tensor boxes = torch::vstack({xyv[1] - anchor_size_y_2, xyv[0] - anchor_size_x_2,
                                           xyv[1] + anchor_size_y_2, xyv[0] + anchor_size_x_2});

      boxes = boxes.swapaxes(0, 1);
      boxes = torch::unsqueeze(boxes, 1); //???
      boxes_level.push_back(boxes);
    }
    torch::Tensor boxes_level_a = torch::concat(boxes_level, 1);

    boxes_all.push_back(boxes_level_a.reshape({-1, 4}));
    boxes_level.clear();
  }

  torch::Tensor anchor_boxes = torch::vstack(boxes_all);

  return anchor_boxes;
}

void draw_boxes(torch::Tensor pred_boxes_x1y1x2y2, std::string image_path, float width, float height)
{

  // pre-processed image is too dark to see, we use original and re-size it

  cv::Mat origImage = cv::imread(image_path);

  cv::Scalar background(124, 116, 104);
  cv::Size dstSize(width,height);
  cv::Mat image_out = resizeKeepAspectRatio(origImage, dstSize, background);

  for (int i = 0; i < pred_boxes_x1y1x2y2.sizes()[0]; i++)
  {
    auto x1 = pred_boxes_x1y1x2y2[i][0].item<float>();
    auto y1 = pred_boxes_x1y1x2y2[i][1].item<float>();
    auto x2 = pred_boxes_x1y1x2y2[i][2].item<float>();
    auto y2 = pred_boxes_x1y1x2y2[i][3].item<float>();

    cv::rectangle(image_out, cv::Point(x1, y1), cv::Point(x2, y2), cv::Scalar(255, 0, 0), 4, 8, 0);
  }
  image_path.replace(image_path.end()-4,image_path.end(),"op.jpg");
  std::cout << "Writing annotated image to " << image_path << std::endl;
  cv::imwrite(image_path,image_out);
}

std::vector<torch::Tensor> postprocess_efficientdet(std::vector<DLTensor *> &tvm_outputs,cv::Size dstSize)
{
  // constants defined in the python code
  float prediction_confidence_threshold = 0.3;
  bool nms_method = false;
  int width = 512;
  int height = 512;
  int num_levels = 5;
  int num_classes = 90;
  int max_detection_points = 5000;
  int max_det_per_image = 15;

  std::vector<torch::Tensor> results;

  // conversion of tvm array to tensor
  std::vector<torch::Tensor> outputs;
  for (DLTensor *pt : tvm_outputs)
  {
    outputs.emplace_back(convert_to_atTensor(pt));
  }

  std::vector<torch::Tensor> scores;
  for (int i = 0; i < 5; i++)
  {
    scores.push_back(outputs[i]);
  }

  std::vector<torch::Tensor> boxes;
  for (int i = 5; i < 10; i++)
  {
    boxes.push_back(outputs[i]);
  }

  int batch_size = scores[0].sizes()[0];

  std::vector<torch::Tensor> to_cat_c;
  for (int level = 0; level < num_levels; level++)
  {
    auto t = scores[level].permute({0, 2, 3, 1}).reshape({batch_size, -1, num_classes});
    to_cat_c.emplace_back(t);
  }
  torch::Tensor cls_outputs_all = torch::cat(to_cat_c, 1);

  std::vector<torch::Tensor> to_cat_b;
  for (int level = 0; level < num_levels; level++)
  {
    auto t = boxes[level].permute({0, 2, 3, 1}).reshape({batch_size, -1, 4});
    to_cat_b.emplace_back(t);
  }
  torch::Tensor box_outputs_all = torch::cat(to_cat_b, 1);

  auto cls_topk_indices_all = torch::topk(cls_outputs_all.reshape({batch_size, -1}), max_detection_points, 1);

  auto indices_all = torch::div(std::get<1>(cls_topk_indices_all), num_classes,"trunc");
  auto classes_all = std::get<1>(cls_topk_indices_all) % num_classes;

  auto box_outputs_all_after_topk = torch::gather(box_outputs_all, 1, indices_all.unsqueeze(2).expand({-1, -1, 4}));

  auto cls_outputs_all_after_topk = torch::gather(cls_outputs_all, 1, indices_all.unsqueeze(2).expand({-1, -1, num_classes}));
  cls_outputs_all_after_topk = torch::gather(cls_outputs_all_after_topk, 2, classes_all.unsqueeze(2));

  torch::Tensor anchors = generate_anchors(dstSize.height, dstSize.width, 3, 7, 3,box_outputs_all_after_topk.device().type());

  torch::Tensor detections;

  for (int i = 0; i < batch_size; i++)
  {
    auto class_out = cls_outputs_all_after_topk[i];
    auto box_out = box_outputs_all_after_topk[i];
    auto indices = indices_all[i];
    auto classes = classes_all[i];


    torch::Tensor anchor_boxes = anchors.index({indices, at::indexing::Slice(0, at::indexing::None)});

    auto box_out_decoded = decode_box_outputs(box_out, anchor_boxes, true);

    auto scores = class_out.sigmoid().squeeze(1); 
    auto result = vision::ops::nms(box_out_decoded,scores,0.45);
    result = result.slice(0,0,max_det_per_image);

    box_out_decoded = box_out_decoded.index({result});
    classes = classes.index({result,torch::indexing::None}) + 1;
    scores = scores.index({result,torch::indexing::None});

    detections = torch::cat({box_out_decoded,scores,classes},1);

    auto filtered_detections = torch::where(detections.index({"...",4}) > prediction_confidence_threshold);
    detections = detections.index({filtered_detections[0]});

    results.emplace_back(detections);
  }
  return results;
}

at::Tensor convert_to_atTensor(DLTensor *tvm_output)
{
  DLManagedTensor *output = new DLManagedTensor{};
  output->dl_tensor = *tvm_output;
  output->deleter = &monly_deleter;
  at::Tensor res = at::fromDLPack(output);
  return res;
}
