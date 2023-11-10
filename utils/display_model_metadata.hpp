
#include <iomanip>
#include <iostream>

void PrintModelMetadata(LRE::LatentRuntimeEngine &model){
  std::cout << std::setw(12) << "----------------------------------------\n";
  std::cout << std::left << std::setw(12) << "Unique ID:\n";
  std::cout << std::setw(12) << model.getModelID() << "\n";
  std::cout << std::setw(12) << "----------------------------------------\n";
  std::cout << std::left << std::setw(12) << "Creation Date:\n";
  std::cout << std::setw(12) << model.getModelMetadata().creation_date << "\n";
  std::cout << std::setw(12) << "----------------------------------------\n";
  std::cout << std::left << std::setw(12) << "Device:\n";
  std::cout << std::setw(12) << model.getDeviceType() << "\n";
  std::cout << std::setw(12) << "----------------------------------------\n";
  std::cout << std::left << std::setw(12) << "Precision: \n";
  std::cout << std::setw(12) << model.getModelPrecision() << "\n";
  std::cout << std::setw(12) << "----------------------------------------\n";
  std::cout << std::left << std::setw(12) << "Input Shapes: \n";
  std::cout << std::setw(12) << model.getInputShapes() << "\n";
  std::cout << std::setw(12) << "----------------------------------------\n";
  std::cout << std::left << std::setw(12) << "Output Shapes: \n";
  std::cout << std::setw(12) << model.getOutputShapes() << "\n";
  std::cout << std::setw(12) << "----------------------------------------\n";
  std::cout << std::left << std::setw(12) << "SDK version: \n";
  std::cout << std::setw(12) << model.getSDKVersion() << "\n";
  std::cout << std::setw(12) << "----------------------------------------\n";
}
