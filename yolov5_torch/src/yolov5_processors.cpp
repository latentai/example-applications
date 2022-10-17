#include "yolov5_processors.hpp"
// #include <sys/time.h>

cv::Mat preprocess_yolov5(cv::Mat &ImageInput, float width, float height) {


  cv::resize(ImageInput, ImageInput, cv::Size(width, height));  // Resize
  cv::cvtColor(ImageInput, ImageInput, cv::COLOR_BGR2RGB);  // RGB Format required
  ImageInput.convertTo(ImageInput, CV_32FC3, 1.f / 255);  // Convert to float ranges 0-1
  cv::dnn::blobFromImage(ImageInput, ImageInput);  // NHWC to NCHW
  return ImageInput;
}

std::vector<at::Tensor> postprocess_yolov5(std::vector<DLTensor *> &tvm_outputs) {

  // struct timeval t0, t1, t2, t3;

  std::vector<at::Tensor> dloutputs,result_output;
  
  // gettimeofday(&t0, 0);
  for (int i = 0; i < tvm_outputs.size() ; i++){

    DLManagedTensor* output = new DLManagedTensor{};
    output->dl_tensor = *tvm_outputs[i];
    output->deleter = &monly_deleter;

    auto op = at::fromDLPack(output);


    if(i<3) // rehshape the Heads
    {
      auto dlTensorShape = op.sizes();
      op = op.reshape({dlTensorShape[0],-1,dlTensorShape[4]});
    }  
    dloutputs.emplace_back(op);
  }

  // gettimeofday(&t1, 0);



  // Decoding 

  auto pred_logits =  at::sigmoid(at::cat({ dloutputs[0],dloutputs[1],dloutputs[2] },1)[0]);
  auto scores = pred_logits.slice(1,5) * pred_logits.slice(1,4,5);
  
  auto pred_wh = (pred_logits.slice(1,0,2) * 2 + dloutputs[3]) * dloutputs[4];
  auto pred_xy = (pred_logits.slice(1,2,4) * 2).pow(2) * dloutputs[5] * 0.5;

  auto pred_wh_unbinded = pred_wh.unbind(-1);
  auto pred_xy_unbinded = pred_xy.unbind(-1);

  auto x1 = pred_wh_unbinded[0] -  pred_xy_unbinded[0];
  auto y1 = pred_wh_unbinded[1] -  pred_xy_unbinded[1];
  auto x2 = pred_wh_unbinded[0] +  pred_xy_unbinded[0];
  auto y2 = pred_wh_unbinded[1] +  pred_xy_unbinded[1];


  auto pred_boxes_x1y1x2y2 = at::stack({x1,y1,x2,y2},-1);

  // Drob boxes below Threshold 

  auto inds_scores = at::where(scores > 0.45);

  pred_boxes_x1y1x2y2 = pred_boxes_x1y1x2y2.index({inds_scores[0]});
  scores = scores.index({inds_scores[0],inds_scores[1]});

  // gettimeofday(&t2, 0);

  //NMS

  auto result = vision::ops::nms(pred_boxes_x1y1x2y2,scores,0.45);

  // gettimeofday(&t3, 0);


  result_output.emplace_back(pred_boxes_x1y1x2y2.index({result}));
  result_output.emplace_back(scores.index({result}));

  // std::cout << std::setprecision(2) << std::fixed;
  // std::cout << "Timing: " << (t1.tv_sec - t0.tv_sec) * 1000 + (t1.tv_usec - t0.tv_usec) / 1000.f << " ms copy to tensors" << std::endl;
  // std::cout << "Timing: " << (t2.tv_sec - t1.tv_sec) * 1000 + (t2.tv_usec - t1.tv_usec) / 1000.f << " ms decoding" << std::endl;
  // std::cout << "Timing: " << (t3.tv_sec - t2.tv_sec) * 1000 + (t3.tv_usec - t2.tv_usec) / 1000.f << " ms nms" << std::endl;
  
  return result_output;


}

