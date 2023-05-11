// ******************************************************************************
// Copyright (c) 2019-2023 by Latent AI Inc. All Rights Reserved.
//
// This file is part of the example-applications (LRE) product,
// and is released under the Apache 2.0 License.
// *****************************************************************************/

#include "imagenet_torch_nchw_processors.hpp"
#include <timer_example.hpp>
#include <cmath>
#include <vector>


cv::Mat preprocess_imagenet_torch_nchw(cv::Mat &ImageInput) {

    cv::cvtColor(ImageInput, ImageInput, cv::COLOR_BGR2RGB);
    ImageInput.convertTo(ImageInput, CV_32FC3, 1.f/255.f); // Normalization between 0-1
    cv::subtract(ImageInput,cv::Scalar(0.485f, 0.456f, 0.406f),ImageInput, cv::noArray(), -1);
    cv::divide(ImageInput,cv::Scalar(0.229f, 0.224f, 0.225f),ImageInput,1,-1);
    
    cv::dnn::blobFromImage( ImageInput, ImageInput ); // convert to nchw 

    return ImageInput;
}


std::vector<std::pair<float,float>> postprocess_top_five(std::vector<DLTensor *> &tvm_outputs, std::vector<int> &output_size)
{
    std::vector<float> fdata(output_size[0]); // Generate Vector for Output
    TVMArrayCopyToBytes(tvm_outputs[0], fdata.data(), output_size[0]); // Copy Data from DL Tensor (Model) to Vector

    // Get Index of Top5 Values
    std::vector<float> idx(tvm_outputs[0]->shape[1]);
    std::iota(idx.begin(), idx.end(), 0);

    std::partial_sort(idx.begin(), idx.begin() + 5, idx.end(),
                  [&](size_t A, size_t B) {
                     return fdata[A] > fdata[B];
                  });

    std::vector<std::pair<float,float>> top_five;

    // Store top 5 Values and Index in a Vector of pairs
    for (int i=0; i<5; i++)
    {
        top_five.push_back(std::make_pair(idx[i],fdata[idx[i]])); 
    }
    return top_five;
}

std::vector<float> softmax(std::vector<float> v) {
    std::vector<float> result;
    float sum = 0.0;

    // Compute the exponential of each element and the sum of all exponentials
    for (float x : v) {
        float exp_x = std::exp(x);
        result.push_back(exp_x);
        sum += exp_x;
    }

    // Normalize the exponentials to get the softmax probabilities
    for (float& x : result) {
        x /= sum;
    }
    return result;
}

std::pair<float,float> postprocess_top_one (std::vector<DLTensor *> &tvm_outputs, std::vector<int> &output_size)
{
    std::vector<float> fdata(tvm_outputs[0]->shape[1]);
    TVMArrayCopyToBytes(tvm_outputs[0], fdata.data(), output_size[0]);
    fdata = softmax(fdata);
    
    int max_element_index = std::max_element(fdata.begin(),fdata.end()) - fdata.begin();
    std::pair<float,float> top_one = std::make_pair(max_element_index,fdata[max_element_index]);
    
    return top_one;
}

void printTopOne(std::pair<float,float> top_one, std::string &label_file_name){
  std::string line;
  std::ifstream label_file;
  label_file.open(label_file_name);
  for (int i = -1; i < top_one.first; i++) {
    std::getline(label_file, line);
  }
  std::cout << " ------------------------------------------------------------ \n Detections \n ------------------------------------------------------------ \n The image prediction result is: id " << top_one.first 
  << "\n Name: " << line << "\n Score: " << top_one.second << "\n ------------------------------------------------------------" << std::endl ;
}

void printTopFive(std::vector<std::pair<float,float>> top_five, std::string &label_file_name){
    std::string line;
    std::ifstream label_file;
    label_file.open(label_file_name);

    for (int i = -1; i < top_five[0].first; i++) {
    std::getline(label_file, line);
    }

    std::cout << " ------------------------------------------------------------ \n Detections \n------------------------------------------------------------ \n The image prediction result is: id" << top_five[0].first 
        << " Name: " << line << " Score: " << top_five[0].second << "\n ------------------------------------------------------------" << std::endl ;
}