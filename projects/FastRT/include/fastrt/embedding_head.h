#pragma once

#include <map>
#include "NvInfer.h"
#include "fastrt/module.h"
#include "fastrt/struct.h"
using namespace nvinfer1;

namespace fastrt {

    class embedding_head : public Module {
    public:
        embedding_head() = default;
        ~embedding_head() = default;

        ILayer* topology(INetworkDefinition *network, 
            std::map<std::string, Weights>& weightMap,
            ITensor& input, 
            const FastreidConfig& reidCfg) override;
    };
}