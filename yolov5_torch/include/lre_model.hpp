#include <dlpack/dlpack.h>
#include <tvm/runtime/module.h>
#include <tvm/runtime/packed_func.h>
#include <tvm/runtime/registry.h>

class LreModel {
 public:
  LreModel() = default;
  LreModel(std::string &t_model_binary_path, DLDevice &t_device);

  int total_outputs, total_inputs, input_width, input_height;

  std::vector<DLTensor *> tvm_inputs;
  std::vector<DLTensor *> tvm_outputs;

  std::vector<int> output_size;
  std::vector<int> input_size;

  void LoadInputs();
  void LoadOutputs();
  void getWidthAndHeight();

  void InferFor(void *t_input_data, int t_iterations);
  void InferOnce(void *t_input_data);

  ~LreModel();

  // Properties:
 private:
  std::string model_binary_path{};
  DLDevice device{};
  tvm::runtime::PackedFunc get_output, set_input, run, get_input,
      get_num_inputs, get_num_outputs;
  tvm::runtime::Module mod_factory, model;
  std::vector<int> PopulateVectors(int &vector_size,
                                   std::vector<DLTensor *> &tvm_vector,
                                   bool input);
  tvm::runtime::Module getModel();
  void getModelFactory();
};
