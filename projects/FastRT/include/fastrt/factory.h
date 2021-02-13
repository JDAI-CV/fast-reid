#pragma once

#include "struct.h"
#include "module.h"
#include "IPoolingLayerRT.h"

namespace fastrt {
    
    class ModuleFactory {
    public:
        ModuleFactory() = default;
        ~ModuleFactory() = default;

        std::unique_ptr<Module> createBackbone(const FastreidBackboneType& backbonetype);
        std::unique_ptr<Module> createHead(const FastreidHeadType& headtype);
    };

    class LayerFactory {
    public:
        LayerFactory() = default;
        ~LayerFactory() = default;

        std::unique_ptr<IPoolingLayerRT> createPoolingLayer(const FastreidPoolingType& pooltype);
    };

}