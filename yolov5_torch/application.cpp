#include <tvm/runtime/latentai/lre_model.hpp>
#include "yolov5_processors.hpp"

#include <sys/time.h>
#include <experimental/filesystem>

namespace fs = std::experimental::filesystem;


int main(int argc, char *argv[]) {
  struct timeval t0, t1, t2, t3;
  
  // Parsing arguments by user
  std::string model_binary{argv[1]};
  std::string image_directory_path{argv[2]};
  std::string key_path = "";
  
  // Model Factory
  DLDevice device_t{kDLCUDA, 0}; //Change to kDLCPU if inference target is a CPU 
  LreModel model(model_binary,key_path, device_t);

  for (const auto & entry : fs::directory_iterator(image_directory_path))
  {
    // Preprocessing
    gettimeofday(&t0, 0);
    std::cout << "Image: " << entry.path() << std::endl;
    cv::Mat image_input = cv::imread(entry.path());
    cv::Mat processed_image =  preprocess_yolov5(image_input,model.input_width,model.input_height);

    // Inference
    gettimeofday(&t1, 0);
    model.InferOnce(processed_image.data);
    gettimeofday(&t2, 0);

    // Post Processing
    auto result = postprocess_yolov5(model.tvm_outputs,entry.path(),model.input_width,model.input_height);

    gettimeofday(&t3, 0);

    std::cout << "pred_boxes_x1y1x2y2 " << std::endl << result[0] << std::endl;
    std::cout << "Scores " << std::endl << result[1] << std::endl;
    std::cout << "Class " << std::endl<< result[2] << std::endl;

    std::cout << std::setprecision(2) << std::fixed;
    std::cout << "Timing: " << (t1.tv_sec - t0.tv_sec) * 1000 + (t1.tv_usec - t0.tv_usec) / 1000.f << " ms pre process" << std::endl;
    std::cout << "Timing: " << (t2.tv_sec - t1.tv_sec) * 1000 + (t2.tv_usec - t1.tv_usec) / 1000.f << " ms infer + copy image" << std::endl;
    std::cout << "Timing: " << (t3.tv_sec - t2.tv_sec) * 1000 + (t3.tv_usec - t2.tv_usec) / 1000.f << " ms post process" << std::endl;
  }
}