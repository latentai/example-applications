/******************************************************************************
 * Copyright (c) 2019-2022 by Latent AI Inc. All Rights Reserved.
 *
 * This file is part of the latentai-lre (LRE) product,
 * and is released under the "Latent AI Commercial Software License".
 *****************************************************************************/

#include <dlpack/dlpack.h>
#include <tvm/runtime/module.h>
#include <tvm/runtime/packed_func.h>
#include <tvm/runtime/registry.h>
#include <fstream>
#include <tvm/runtime/latentai/utils/json_parser.hpp>
#include <tvm/runtime/latentai/lre_cryption_service.hpp>


using json = nlohmann::json;

class LreModel {
  // Properties:
 private:
  std::string model_binary_path{};
  std::string key_path{};
  std::string metadata_path{};
  tvm::runtime::PackedFunc get_output, set_input, run, get_input,
      get_num_inputs, get_num_outputs;
  tvm::runtime::Module mod_factory, model;
  std::vector<int> PopulateVectors(int &vector_size,
                                   std::vector<DLTensor *> &tvm_vector,
                                   bool input);
  tvm::runtime::Module getModel();
  void getModelFactory();
  bool isEncrypted(std::string &source_file);
  bool DecryptModel(std::vector<unsigned char> &t_key);
  void handleModelEncryption(std::vector<unsigned char> &t_key);
 
 public:

/*!
 * \brief Instantiate an LRE Model from a modelLibrary.so generated from LEIP SDK
 */
  LreModel() = default;

/*!
 * \brief Instantiate an LRE Model from a modelLibrary.so generated from LEIP SDK
 * \param t_model_binary_path Path to modelLibray.so
 * \param t_device Device type i.e CPU, GPU
 */
  LreModel(std::string &t_model_binary_path, DLDevice &t_device);
/*!
 * \brief Instantiate an LRE Model from a modelLibrary.so generated from LEIP SDK
 * \param t_model_binary_path Path to modelLibray.so
 * \param t_metadata_path Path to json contatining model requirements like device type
 */
  LreModel(std::string &t_model_binary_path, std::string &t_metadata_path);
/*!
 * \brief Instantiate an LRE Model from an encrypted modelLibrary.so generated from LEIP SDK
 * \param t_model_binary_path Path to encrypted modelLibray.so
 * \param t_key Key used for encryption of modelLibrary.so
 * \param t_device Device type i.e CPU, GPU
 */
  LreModel(std::string &t_model_binary_path, std::vector<unsigned char> &t_key, DLDevice &t_device);
/*!
 * \brief Instantiate an LRE Model from an encrypted modelLibrary.so generated from LEIP SDK
 * \param t_model_binary_path Path to encrypted modelLibray.so
 * \param t_key Key used for encryption of modelLibrary.so
 * \param t_metadata_path Path to json contatining model requirements like device type
 */
  LreModel(std::string &t_model_binary_path, std::vector<unsigned char> &t_key, std::string &t_metadata_path);
  
/*!
 * \var total_outputs The number of outputs returned by the model graph
 * \var total_inputs The number of inputs accepted by the model graph
 * \var input_width Width of the image input[0] for a vision model 
 * \var input_height Height of the image input[0] for a vision model 
 */
  int total_outputs, total_inputs, input_width, input_height;

/*!
 * \var input_widths Vector of widths of the image inputs for a vision model 
 * \var input_heights Vector of height of the image inputs for a vision model 
 */
  std::vector<int> input_widths;
  std::vector<int> input_heights;

/*!
 * \var tvm_inputs Vector of memory allocated DLTensor* for inputs. 
 * \var tvm_outputs Vector of memory allocated DLTensor* for outputs. 
 */
  std::vector<DLTensor *> tvm_inputs;
  std::vector<DLTensor *> tvm_outputs;

/*!
 * \var input_size Vector of sizes of memory for inputs. 
 * \var output_size Vector of sizes of memory for outputs. 
 */
  std::vector<int> output_size;
  std::vector<int> input_size;

/*!
 * \brief Perform multiple inference on the same input. Converts an array to DLTensor*
 * \param t_input_data Pointer to input data. C Array, vector.data(), cv::mat.data etc 
 * \param t_iterations Number of inference to be repeated
 */
  void InferFor(void *t_input_data, int t_iterations);
/*!
 * \brief Perform single inference. Converts an array to DLTensor*. 
 * \param t_input_data Pointer to input data. C Array, vector.data(), cv::mat.data etc 
 */
  void InferOnce(void *t_input_data);
/*!
 * \brief Perform single inference . Converts an array to DLTensor*
 * \param t_input_data Pointer to input data. C Array, vector.data(), cv::mat.data etc 
 */
  void Infer(void *t_input_data);
/*!
 * \brief Perform inference. Converts an array to DLTensor*
 * \param t_input_data Pointer to a DLtensor
 */
  void Infer(DLTensor *t_input_data);
/*!
 * \brief Perform inference. For model graphs with multiple inputs
 * \param t_input_data Vector of pointer to a DLtensor
 */
  void Infer(std::vector<DLTensor *> &tvm_input_data);
  /*!
 * \brief Perform inference. For model graphs with multiple inputs. Converts array to DLTensor*
 * \param t_input_data Vector of pointer to void*
 */
  void Infer(std::vector<void *> &tvm_input_data);

  void Run();
  void LoadInputs();
  void LoadOutputs();
  void GetWidthAndHeight();
  DLDevice GetDevice(std::string &metadata_path);
  DLDevice device{};

  ~LreModel();

};
