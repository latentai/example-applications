// ******************************************************************************
// Copyright (c) 2019-2023 by Latent AI Inc. All Rights Reserved.
//
// This file is part of the example-applications (LRE) product,
// and is released under the Apache 2.0 License.
// *****************************************************************************/

#include "imagenet_torch_nchw_processors.hpp"


cv::Mat preprocess_imagenet_torch_nchw(cv::Mat &ImageInput, float image_shape_height,float image_shape_width) {

    const cv::Size image_size = cv::Size( image_shape_width, image_shape_height );
    cv::cvtColor(ImageInput, ImageInput, cv::COLOR_BGR2RGB);
    cv::resize(ImageInput, ImageInput, image_size,0,0,cv::INTER_NEAREST); 
    ImageInput.convertTo(ImageInput, CV_32FC3, 1.f/255.f); // Normalization between 0-1
    cv::subtract(ImageInput,cv::Scalar(0.485f, 0.456f, 0.406f),ImageInput, cv::noArray(), -1);
    cv::divide(ImageInput,cv::Scalar(0.229f, 0.224f, 0.225f),ImageInput,1,-1);
    
    cv::dnn::blobFromImage( ImageInput, ImageInput ); // convert to nchw 

    return ImageInput;
}


std::vector<std::pair<float,float>> postprocess_top_five(std::vector<DLTensor *> &tvm_outputs, std::vector<int> &output_size, std::string &label_file_name)
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

    std::string line;
    std::ifstream label_file;
    label_file.open(label_file_name);

    for (int i = -1; i < top_five[0].first; i++) {
    std::getline(label_file, line);
    }

    std::cout << " ------------------------------------------------------------ \n Detections \n------------------------------------------------------------ \n The image prediction result is: id" << top_five[0].first 
        << " Name: " << line << " Score: " << top_five[0].second << "\n ------------------------------------------------------------" << std::endl ;
  
    return top_five;
}

std::pair<float,float> postprocess_top_one (std::vector<DLTensor *> &tvm_outputs, std::vector<int> &output_size, std::string &label_file_name)
{

    std::vector<float> fdata(tvm_outputs[0]->shape[1]);
    TVMArrayCopyToBytes(tvm_outputs[0], fdata.data(), output_size[0]);
    
    int max_element_index = std::max_element(fdata.begin(),fdata.end()) - fdata.begin();

    std::pair<float,float> top_one = std::make_pair(max_element_index,fdata[max_element_index]);

    std::string line;
    std::ifstream label_file;
    label_file.open(label_file_name);

    for (int i = -1; i < top_one.first; i++) {
    std::getline(label_file, line);
    }

    std::cout << " \n------------------------------------------------------------ \n Detections \n------------------------------------------------------------ \n The image prediction result is: id " << top_one.first 
        << " Name: " << line << " Score: " << top_one.second << "\n ------------------------------------------------------------" << std::endl ;
  
    return top_one;

}