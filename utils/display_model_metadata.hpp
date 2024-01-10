
#include <iomanip>
#include <iostream>

void PrintModelMetadata(LRE::LatentRuntimeEngine &lre){
  std::cout << std::setw(12) << "----------------------------------------\n";
  std::cout << std::left << std::setw(12) << "Unique ID:\n";
  std::cout << std::setw(12) << lre.getModelID() << "\n";
  std::cout << std::setw(12) << "----------------------------------------\n";
  std::cout << std::left << std::setw(12) << "Creation Date:\n";
  std::cout << std::setw(12) << lre.getModelMetadata().creation_date << "\n";
  std::cout << std::setw(12) << "----------------------------------------\n";
  std::cout << std::left << std::setw(12) << "Device:\n";
  std::cout << std::setw(12) << lre.getDeviceType() << "\n";
  std::cout << std::setw(12) << "----------------------------------------\n";
  std::cout << std::left << std::setw(12) << "Precision: \n";
  std::cout << std::setw(12) << lre.getModelPrecision() << "\n";
  std::cout << std::setw(12) << "----------------------------------------\n";
  std::cout << std::left << std::setw(12) << "Input Shapes: \n";
  std::cout << std::setw(12) << lre.getInputShapes() << "\n";
  std::cout << std::setw(12) << "----------------------------------------\n";
  std::cout << std::left << std::setw(12) << "Output Shapes: \n";
  std::cout << std::setw(12) << lre.getOutputShapes() << "\n";
  std::cout << std::setw(12) << "----------------------------------------\n";
  std::cout << std::left << std::setw(12) << "SDK version: \n";
  std::cout << std::setw(12) << lre.getSDKVersion() << "\n";
  std::cout << std::setw(12) << "----------------------------------------\n";
  std::cout << std::left << std::setw(12) << "is TRT: \n";
  std::cout << std::setw(12) << lre.isTRT() << "\n";
  std::cout << std::setw(12) << "----------------------------------------\n";
}
