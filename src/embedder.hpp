#pragma once
#include <cstdint>
#include <onnxruntime_cxx_api.h>
#include <string>
#include <iostream>
#include <vector>
#include <cmath>
#include "tokenizer/tokenizer.hpp"

class ModelEngine {
    private:
    // ONNX lifecycle objects
    Ort::Env env;
    Ort::SessionOptions session_options;
    Ort::Session* session;

    public:
    // Constructor
    ModelEngine(const std::string& model_path)
        : env(ORT_LOGGING_LEVEL_WARNING, "NexusML") {
        session_options.SetIntraOpNumThreads(1);
        session = new Ort::Session(env, model_path.c_str(), session_options);

        std::cout << "[NEXUS] ONNX Model loaded successfully from: " << model_path << std::endl;
    }

    // Destructor
    ~ModelEngine() {
        delete session;
        std::cout << "[NEXUS] ONNX Session closed safely." << std::endl;
    }

    std::vector<float> generate_embedding(const Encoding& enc){
        std::vector<int64_t> token_type_ids(enc.input_ids.size(),0);

        Ort::MemoryInfo mem_info = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);

        std::vector<int64_t> tensor_shape = {1, static_cast<int64_t>(enc.input_ids.size())};

        Ort::Value input_tensor = Ort::Value::CreateTensor<int64_t>(
            mem_info,
            const_cast<int64_t*>(enc.input_ids.data()),
            enc.input_ids.size(),
            tensor_shape.data(),
            tensor_shape.size()
        );

        Ort::Value attention_mask_tensor = Ort::Value::CreateTensor<int64_t>(
            mem_info,
            const_cast<int64_t*>(enc.attention_mask.data()),
            enc.attention_mask.size(),
            tensor_shape.data(),
            tensor_shape.size()
        );

        Ort::Value token_type_ids_tensor = Ort::Value::CreateTensor<int64_t>(
            mem_info,
            const_cast<int64_t*>(token_type_ids.data()),
            token_type_ids.size(),
            tensor_shape.data(),
            tensor_shape.size()
        );

        std::vector<Ort::Value> input_tensors;
        input_tensors.push_back(std::move(input_tensor));
        input_tensors.push_back(std::move(attention_mask_tensor));
        input_tensors.push_back(std::move(token_type_ids_tensor));

        const char* input_names[] = {"input_ids", "attention_mask", "token_type_ids"};
        const char* output_names[] = {"last_hidden_state"};

        std::vector<Ort::Value> output_tensors = session->Run(
            Ort::RunOptions{nullptr},
            input_names,
            input_tensors.data(),
            input_tensors.size(),
            output_names,
            1
        );

        auto& output_tensor = output_tensors.front();
        float* float_array = output_tensor.GetTensorMutableData<float>();
        size_t count = output_tensor.GetTensorTypeAndShapeInfo().GetElementCount();

        int seq_len = count/384;
        std::vector<float> pooled_embedding(384,0.0f);

        for(int i=0; i<seq_len; i++){
            for(int j=0; j<384; j++){
                pooled_embedding[j] += float_array[(i*384)+j];
            }
        }

        float len = 0;
        for(int i=0; i<384; i++){
            pooled_embedding[i] /= seq_len;
            len += pow(pooled_embedding[i], 2);
        }
        len = sqrt(len);
        for(int i=0; i<384; i++){
            pooled_embedding[i] /= len;
        }
        return pooled_embedding;
    }
};
