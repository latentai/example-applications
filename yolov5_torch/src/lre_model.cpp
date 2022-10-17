#include "lre_model.hpp"

LreModel::LreModel(std::string &t_model_binary_path, DLDevice &t_device)
    : model_binary_path(t_model_binary_path), device(t_device) {
  getModelFactory();

  get_input = mod_factory.GetFunction("get_input");
  set_input = mod_factory.GetFunction("set_input");
  get_num_inputs = mod_factory.GetFunction("get_num_inputs");
  get_output = mod_factory.GetFunction("get_output");
  get_num_outputs = mod_factory.GetFunction("get_num_outputs");

  total_outputs = get_num_outputs();
  total_inputs = get_num_inputs();

  LoadInputs();
  LoadOutputs();
  getWidthAndHeight();

  run = mod_factory.GetFunction("run");
};

void LreModel::LoadInputs() {
  // LoadInputs();
  input_size = PopulateVectors(total_inputs, tvm_inputs, 1);
  // Print(tvm_inputs);
};
void LreModel::LoadOutputs() {
  // LoadInputs();
  output_size = PopulateVectors(total_outputs, tvm_outputs, 0);
  // Print(tvm_outputs);
};

void LreModel::getWidthAndHeight() {
  DLTensor *tvm_array{};

  tvm_array = get_input(0);
  if(tvm_array->ndim > 3)
  {
    if(tvm_array->shape[1] > 3)
    {
      input_width = tvm_array->shape[2];
      input_height = tvm_array->shape[1];
    }
    else{
      input_width = tvm_array->shape[3];
      input_height = tvm_array->shape[2];
    }
  }
  else{

    input_width = 1;
    input_height = 1;
  }

}

tvm::runtime::Module LreModel::getModel() {
  return tvm::runtime::Module::LoadFromFile(model_binary_path);
};

void LreModel::getModelFactory() {
  // Load the model binary
  model = getModel();
  // Create the Graph and Executor Module
  mod_factory = model.GetFunction("default")(device);
};

static inline size_t GetDataSize(const DLTensor *t) {
  size_t size = 1;
  for (tvm_index_t i = 0; i < t->ndim; ++i) {
    size *= t->shape[i];
  }
  size *= (t->dtype.bits * t->dtype.lanes + 7) / 8;
  // auto ip_size{sizeof(tvmarray->dtype)}; // 4
  return size;
};

std::vector<int> LreModel::PopulateVectors(int &vector_size,
                                           std::vector<DLTensor *> &tvm_vector,
                                           bool input) {
  std::vector<int> sizes{};
  DLTensor *tvmarray{};
  DLTensor *newObj{};

  for (int i = 0; i < vector_size; i++) {
    tvmarray = tvm_vector[i];
    if (input) {
      tvmarray = get_input(i);
    } else {
      tvmarray = get_output(i);
    }

    size_t ip_size{GetDataSize(tvmarray)};
    sizes.emplace_back(ip_size);

    DLTensor *newObj = new DLTensor();
    tvm_vector.emplace_back(newObj);
    TVMArrayAlloc(tvmarray->shape, tvmarray->ndim, tvmarray->dtype.code,
                  tvmarray->dtype.bits, tvmarray->dtype.lanes,
                  tvmarray->device.device_type, tvmarray->device.device_id,
                  &(tvm_vector[i]));
  }

  return sizes;
};

void LreModel::InferOnce(void *t_input_data) {
  TVMArrayCopyFromBytes(tvm_inputs[0], t_input_data, input_size[0]);
  set_input(0, tvm_inputs[0]);

  run();

  for (int i = 0; i < total_outputs; i++) {
    get_output(i, tvm_outputs[i]);
  }
};

void LreModel::InferFor(void *t_input_data, int t_iterations) {
  TVMArrayCopyFromBytes(tvm_inputs[0], t_input_data, input_size[0]);
  set_input(0, tvm_inputs[0]);

  // Run for iterations
  for (int i = 0; i < t_iterations; i++) {
    run();
    for (int j = 0; j < total_outputs; j++) {
      get_output(j, tvm_outputs[j]);
    }
  }
};

LreModel::~LreModel() {
  // std::cout << "object was destructed" << std::endl;
}
