#pragma once

#include <map>
#include "NvInfer.h"
#include "fastrt/struct.h"
using namespace nvinfer1;

namespace fastrt {

    IScaleLayer* embedding_head(INetworkDefinition *network, 
        std::map<std::string, Weights>& weightMap, 
        ITensor& input, 
        const FastreidConfig& reidCfg);

}