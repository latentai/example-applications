#include <opencv2/imgproc.hpp>
#include <opencv2/opencv.hpp>

cv::Mat ReadImage(const std::string &img_path){
    cv::Mat cv_image = cv::imread(img_path);
    return cv_image;
}

cv::Mat ResizeImage(cv::Mat &image, float width, float height){
    const cv::Size image_size = cv::Size(width, height);
    cv::resize(image, image, image_size,0,0,cv::INTER_NEAREST); 
    return image;
}