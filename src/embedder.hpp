#pragma once
#include "tokenizer/tokenizer.hpp"
#include <cmath>
#include <cstdint>
#include <iostream>
#include <onnxruntime_cxx_api.h>
#include <string>
#include <vector>

class VecFSEngine {
private:
  // ONNX lifecycle objects
  Ort::Env env; // Global Environment, manages master thread pools, error
                // logging, global state
  Ort::SessionOptions session_options; // Configuration struct
  Ort::Session *session; // Will have the actual model loaded into RAM

public:
  // Constructor
  VecFSEngine(const std::string
                  &model_path) // : env(...) is an initialiser list, forcing the
                               // env object to be created with tose args before
                               // the constructor runs
      : env(ORT_LOGGING_LEVEL_WARNING, "VecFSML") {

    // Since the daemon runs in background, we force it to use only 1 CPU
    // thread. Slows embedding but ensures no lag
    session_options.SetIntraOpNumThreads(1);

    // Create a session, load the model and the options and env
    session = new Ort::Session(env, model_path.c_str(), session_options);

    std::cout << "[VecFS] ONNX Model loaded successfully from: " << model_path
              << std::endl;
  }

  // Destructor
  ~VecFSEngine() {
    delete session;
    std::cout << "[VecFS] ONNX Session closed safely." << std::endl;
  }

  std::vector<float> generate_embedding(const Encoding &enc) {

    // Text models expect 3 inputs, input_ids, attention_mask and token_type_ids
    // The third is used for Q&A tasks to separate Q's from A's
    // We don't need it here so we just make a dummy vector filled with 0's
    std::vector<int64_t> token_type_ids(enc.input_ids.size(), 0);

    // Tell ORT to use the default mem allocator as the data is in standard cpu
    // ram. If it were on GPU we'd allocate VRAM on the GPU instead
    Ort::MemoryInfo mem_info =
        Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);

    // Neural networks expect 2d or 3d shape tensors so we define a abtch size
    // of 1
    std::vector<int64_t> tensor_shape = {
        1, static_cast<int64_t>(enc.input_ids.size())};

    // Wrap the vector into an ORT Tensor
    Ort::Value input_tensor = Ort::Value::CreateTensor<int64_t>(
        mem_info, const_cast<int64_t *>(enc.input_ids.data()),
        enc.input_ids.size(), tensor_shape.data(), tensor_shape.size());

    // Repeat for attention mask
    Ort::Value attention_mask_tensor = Ort::Value::CreateTensor<int64_t>(
        mem_info, const_cast<int64_t *>(enc.attention_mask.data()),
        enc.attention_mask.size(), tensor_shape.data(), tensor_shape.size());

    // Repeat for token type ids
    Ort::Value token_type_ids_tensor = Ort::Value::CreateTensor<int64_t>(
        mem_info, const_cast<int64_t *>(token_type_ids.data()),
        token_type_ids.size(), tensor_shape.data(), tensor_shape.size());

    // Push all the individual tensors into a vector to pass as a group.
    // std::move avoids creating a copy of the tensor
    std::vector<Ort::Value> input_tensors;
    input_tensors.push_back(std::move(input_tensor));
    input_tensors.push_back(std::move(attention_mask_tensor));
    input_tensors.push_back(std::move(token_type_ids_tensor));

    // These are te required names
    const char *input_names[] = {"input_ids", "attention_mask",
                                 "token_type_ids"};
    const char *output_names[] = {"last_hidden_state"};

    // Run the model
    std::vector<Ort::Value> output_tensors = session->Run(
        Ort::RunOptions{nullptr}, // No special runtime rules
        input_names, input_tensors.data(), input_tensors.size(), output_names,
        1 // Number of output tensors received
    );

    // Get the first tensor, extract the float values and find count
    auto &output_tensor = output_tensors.front();
    float *float_array = output_tensor.GetTensorMutableData<float>();
    size_t count = output_tensor.GetTensorTypeAndShapeInfo().GetElementCount();

    // Niw we do Mean Pooling and L2 Norm

    // Find out number of words in sequence, create an empty pooled embedding
    int seq_len = count / 384;
    std::vector<float> pooled_embedding(384, 0.0f);

    // Fill up the pooled embedding by averaging out the embeddings of every
    // word
    for (int i = 0; i < seq_len; i++) {
      for (int j = 0; j < 384; j++) {
        pooled_embedding[j] += float_array[(i * 384) + j];
      }
    }

    // Normalize the embedding vector to have magnitude of 1.0
    float len = 0;
    for (int i = 0; i < 384; i++) {
      pooled_embedding[i] /= seq_len;
      len += pow(pooled_embedding[i], 2);
    }
    len = sqrt(len);
    for (int i = 0; i < 384; i++) {
      pooled_embedding[i] /= len;
    }

    // Return the final sentence embedding
    return pooled_embedding;
  }

  // Generate embeddings for images
  std::vector<float>
  generate_image_embedding(const std::vector<float> &norm_img) {

    // Memory allocation in ram
    Ort::MemoryInfo mem_info =
        Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);

    // Hardcode the tensor shape
    std::vector<int64_t> tensor_shape = {1, 3, 224, 224};

    // Create the input tensor
    Ort::Value input_tensor = Ort::Value::CreateTensor<float>(
        mem_info, const_cast<float *>(norm_img.data()), norm_img.size(),
        tensor_shape.data(), tensor_shape.size());

    // Prepare the input tensors vector
    std::vector<Ort::Value> input_tensors;
    input_tensors.push_back(std::move(input_tensor));

    // The field names
    const char *input_names[] = {"pixel_values"};
    const char *output_names[] = {"image_embeds"};

    // Generate the embedding
    std::vector<Ort::Value> output_tensors = session->Run(
        Ort::RunOptions{nullptr}, input_names, input_tensors.data(),
        input_tensors.size(), output_names, 1);

    // Extract the float pointer
    auto &output_tensor = output_tensors.front();
    float *float_array = output_tensor.GetTensorMutableData<float>();
    size_t count = output_tensor.GetTensorTypeAndShapeInfo().GetElementCount();

    // Convert it to an embedding vector and return it.
    std::vector<float> embedding(float_array, float_array + count);
    return embedding;
  }
};
