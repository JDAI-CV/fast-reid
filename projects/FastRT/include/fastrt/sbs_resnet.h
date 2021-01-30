#pragma once

#include <map>
#include <functional>
#include "struct.h"
#include "NvInfer.h"
using namespace nvinfer1;

namespace fastrt {

    IActivationLayer* backbone_sbsR34_distill(INetworkDefinition *network, 
        std::map<std::string, Weights>& weightMap, 
        ITensor& input,
        const FastreidConfig& reidCfg);   

    IActivationLayer* backbone_sbsR50_distill(INetworkDefinition *network, 
        std::map<std::string, Weights>& weightMap, 
        ITensor& input,
        const FastreidConfig& reidCfg); 

    IActivationLayer* backbone_sbsR34(INetworkDefinition *network, 
        std::map<std::string, Weights>& weightMap, 
        ITensor& input,
        const FastreidConfig& reidCfg);

    IActivationLayer* backbone_sbsR50(INetworkDefinition *network, 
        std::map<std::string, Weights>& weightMap, 
        ITensor& input,
        const FastreidConfig& reidCfg);

}

namespace fastrt {

    template <typename T>
    using backboneFcn = std::function<T*(INetworkDefinition*, std::map<std::string, Weights>&, ITensor&, const FastreidConfig&)>;

    template <typename T>
    backboneFcn<T> createBackbone(const FastreidConfig &reidcfg) {
        switch(reidcfg.backbone) {
            case FastreidBackboneType::r50:   
                /* cfg.MODEL.META_ARCHITECTURE: Baseline */  
                /* cfg.MODEL.BACKBONE.DEPTH: 50x */ 
                std::cout << "[CreateBackbone]: backbone_sbsR50" << std::endl;
                return backbone_sbsR50;
                break;
            case FastreidBackboneType::r50_distill: 
                /* cfg.MODEL.META_ARCHITECTURE: Distiller */ 
                /* cfg.MODEL.BACKBONE.DEPTH: 50x */   
                std::cout << "[CreateBackbone]: backbone_sbsR50_distill" << std::endl;
                return backbone_sbsR50_distill;
                break;
            case FastreidBackboneType::r34: 
                /* cfg.MODEL.META_ARCHITECTURE: Baseline */  
                /* cfg.MODEL.BACKBONE.DEPTH: 34x */  
                std::cout << "[CreateBackbone]: backbone_sbsR34" << std::endl;
                return backbone_sbsR34;
                break;
            case FastreidBackboneType::r34_distill: 
                /* cfg.MODEL.META_ARCHITECTURE: Distiller */ 
                /* cfg.MODEL.BACKBONE.DEPTH: 34x */  
                std::cout << "[CreateBackbone]: backbone_sbsR34_distill" << std::endl;
                return backbone_sbsR34_distill;
                break;
            default:
                std::cout << "[Backbone is not supported.]" << std::endl;
        }
        return nullptr;
    }

}