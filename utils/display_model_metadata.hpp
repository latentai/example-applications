
#include <iomanip>
#include <iostream>

void PrintModelMetadata(LreModel &model){
  std::cout << std::setw(8) << "----------------------------------------\n";
  std::cout << std::left << std::setw(8) <<  "Device" << " | " << "Flag    " << " | " << "UUID\n";
  std::cout << std::setw(8) << "--------" << " | " << "--------" << " | " << "------------------\n";
  std::cout << std::setw(8) << getEnumString(model.GetModelMetadata().device) << " | " <<  getEnumString(model.GetModelMetadata().flag) << "     | " << model.GetModelMetadata().uuid << "\n";
  std::cout << std::setw(8) << "----------------------------------------\n";
}
