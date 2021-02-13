#pragma once

#include "struct.h"
#include "module.h"

namespace fastrt {
    
    class ModuleFactory {
    public:
        ModuleFactory() = default;
        ~ModuleFactory() = default;

        std::unique_ptr<Module> createBackbone(const FastreidBackboneType& backbonetype);
        std::unique_ptr<Module> createHead(const FastreidHeadType& headtype);
    };

}