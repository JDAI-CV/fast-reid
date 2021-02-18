#include <iostream>
#include "fastrt/utils.h"
#include "fastrt/sbs_resnet.h"
#include "fastrt/factory.h"
#include "fastrt/embedding_head.h"
#include "../layers/poolingLayerRT.h"

namespace fastrt {

    std::unique_ptr<Module> ModuleFactory::createBackbone(const FastreidBackboneType& backbonetype) {
        switch(backbonetype) {
            case FastreidBackboneType::r50:   
                /* cfg.MODEL.META_ARCHITECTURE: Baseline */  
                /* cfg.MODEL.BACKBONE.DEPTH: 50x */ 
                std::cout << "[createBackboneModule]: backbone_sbsR50" << std::endl;
                return make_unique<backbone_sbsR50>();
            case FastreidBackboneType::r50_distill: 
                /* cfg.MODEL.META_ARCHITECTURE: Distiller */ 
                /* cfg.MODEL.BACKBONE.DEPTH: 50x */   
                std::cout << "[createBackboneModule]: backbone_sbsR50_distill" << std::endl;
                return make_unique<backbone_sbsR50_distill>();
            case FastreidBackboneType::r34: 
                /* cfg.MODEL.META_ARCHITECTURE: Baseline */  
                /* cfg.MODEL.BACKBONE.DEPTH: 34x */  
                std::cout << "[createBackboneModule]: backbone_sbsR34" << std::endl;
                return make_unique<backbone_sbsR34>();
            case FastreidBackboneType::r34_distill: 
                /* cfg.MODEL.META_ARCHITECTURE: Distiller */ 
                /* cfg.MODEL.BACKBONE.DEPTH: 34x */  
                std::cout << "[createBackboneModule]: backbone_sbsR34_distill" << std::endl;
                return make_unique<backbone_sbsR34_distill>();
            default:
                std::cerr << "[Backbone is not supported.]" << std::endl;
                return nullptr;
        }
    }

    std::unique_ptr<Module> ModuleFactory::createHead(const FastreidHeadType& headtype) {
        switch(headtype) {
            case FastreidHeadType::EmbeddingHead:   
                /* cfg.MODEL.HEADS.NAME: EmbeddingHead */ 
                std::cout << "[createHeadModule]: EmbeddingHead" << std::endl;
                return make_unique<embedding_head>();
            default:
                std::cerr << "[Head is not supported.]" << std::endl;
                return nullptr;
        }
    }

    std::unique_ptr<IPoolingLayerRT> LayerFactory::createPoolingLayer(const FastreidPoolingType& pooltype) {
        switch(pooltype) {
            case FastreidPoolingType::maxpool:
                std::cout << "[createPoolingLayer]: maxpool" << std::endl;
                return make_unique<MaxPool>();
            case FastreidPoolingType::avgpool:
                std::cout << "[createPoolingLayer]: avgpool" << std::endl;
                return make_unique<AvgPool>();
            case FastreidPoolingType::gempool:
                std::cout << "[createPoolingLayer]: gempool" << std::endl;
                return make_unique<GemPool>();
            case FastreidPoolingType::gempoolP:
                std::cout << "[createPoolingLayer]: gempoolP" << std::endl;
                return make_unique<GemPoolP>();
            default:
                std::cerr << "[Pooling layer is not supported.]" << std::endl; 
                return nullptr;
        }  
    }

}