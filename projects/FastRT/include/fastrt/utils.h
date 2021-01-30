#pragma once

#include <map>
#include <chrono>
#include <memory>
#include <vector>
#include <fstream>
#include <iostream>

#include "assert.h"
#include "NvInfer.h"
#include "cuda_runtime_api.h"
#include "fastrt/struct.h"

#define CHECK(status)                             \
    do                                            \
    {                                             \
        auto ret = (status);                      \
        if (ret != 0)                             \
        {                                         \
            std::cout << "Cuda failure: " << ret; \
            abort();                              \
        }                                         \
    } while (0)


#define TRTASSERT(CONDITION)                      \
    do                                            \
    {                                             \
        auto cond = (CONDITION);                  \
        if (!cond)                                \
        {                                         \
            std::cerr << "Condition failure.\n";  \
            abort();                              \
        }                                         \
    } while (0)


using Time = std::chrono::high_resolution_clock;
using TimePoint = std::chrono::time_point<std::chrono::high_resolution_clock>;

template<typename T, typename... Args>
std::unique_ptr<T> make_unique(Args&&... args) {
    return std::unique_ptr<T>(new T(std::forward<Args>(args)...));
}

namespace io {
    std::vector<std::string> fileGlob(const std::string& pattern);
}

namespace trt {
    /* 
     * Load weights from files shared with TensorRT samples.
     * TensorRT weight files have a simple space delimited format:
     * [type] [size] <data x size in hex>
     */ 
    std::map<std::string, nvinfer1::Weights> loadWeights(const std::string file);

    std::ostream& operator<<(std::ostream& os, const ModelConfig& modelCfg);
}

namespace fastrt {
    std::ostream& operator<<(std::ostream& os, const FastreidConfig& fastreidCfg);
}