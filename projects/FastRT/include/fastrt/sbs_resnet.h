#pragma once

#include <map>
#include "struct.h"
#include "module.h"
#include "NvInfer.h"
using namespace nvinfer1;

namespace fastrt {

    class backbone_sbsR34_distill : public Module {
    public:
        backbone_sbsR34_distill() = default;
        ~backbone_sbsR34_distill() = default;
        ILayer* topology(INetworkDefinition *network, 
            std::map<std::string, Weights>& weightMap, 
            ITensor& input,
            const FastreidConfig& reidCfg) override; 
    };

    class backbone_sbsR50_distill : public Module {  
    public:
        backbone_sbsR50_distill() = default;   
        ~backbone_sbsR50_distill() = default;
        ILayer* topology(INetworkDefinition *network, 
            std::map<std::string, Weights>& weightMap, 
            ITensor& input,
            const FastreidConfig& reidCfg) override;
    };

    class backbone_sbsR34 : public Module {
    public:
        backbone_sbsR34() = default;
        ~backbone_sbsR34() = default;
        ILayer* topology(INetworkDefinition *network, 
            std::map<std::string, Weights>& weightMap, 
            ITensor& input,
            const FastreidConfig& reidCfg) override;
    };

    class backbone_sbsR50 : public Module {
    public:
        backbone_sbsR50() = default;
        ~backbone_sbsR50() = default;
        ILayer* topology(INetworkDefinition *network, 
            std::map<std::string, Weights>& weightMap, 
            ITensor& input,
            const FastreidConfig& reidCfg) override;
    };
     
}