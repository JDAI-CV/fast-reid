#pragma once

#include "utils.h"
#include "holder.h"
#include "layers.h"
#include "struct.h"
#include "InferenceEngine.h"

#include <memory>
#include <vector>
#include <opencv2/opencv.hpp>
extern Logger gLogger;
using namespace trt;
using namespace trtxapi;

namespace fastrt {

    class Model {
    public:
        Model(const trt::ModelConfig &modelcfg, 
            const FastreidConfig &reidcfg, 
            const std::string input_name="input", 
            const std::string output_name="output");

        virtual ~Model() {};

        template <typename B, typename H>
        bool serializeEngine(const std::string engine_file, 
            std::function<B*(INetworkDefinition*, std::map<std::string, Weights>&, ITensor&, const FastreidConfig&)> backbone,
            std::function<H*(INetworkDefinition*, std::map<std::string, Weights>&, ITensor&, const FastreidConfig&)> head) {

            /* Create builder */  
            auto builder = make_holder(createInferBuilder(gLogger));

            /* Create model to populate the network, then set the outputs and create an engine */ 
            auto engine = createEngine<B, H>(builder.get(), backbone, head);
            TRTASSERT(engine.get());

            /* Serialize the engine */ 
            auto modelStream = make_holder(engine->serialize());
            TRTASSERT(modelStream.get());

            std::ofstream p(engine_file, std::ios::binary | std::ios::out);
            if (!p) {
                std::cerr << "could not open plan output file" << std::endl;
                return false;
            }
            p.write(reinterpret_cast<const char*>(modelStream->data()), modelStream->size());
            std::cout << "[Save serialized engine]: " << engine_file << std::endl;
            return true;
        }

        bool deserializeEngine(const std::string engine_file);

        /* Support batch inference */
        bool inference(std::vector<cv::Mat> &input); 

        /* 
         * Access the memory allocated by cudaMallocHost. (It's on CPU side) 
         * Use this after each inference.
         */ 
        float* getOutput(); 

        /* 
         * Output buffer size
         */ 
        int getOutputSize(); 

        /* 
         * Cuda device id
         * You may need this in multi-thread/multi-engine inference
         */ 
        int getDeviceID(); 

    private:
        template <typename B, typename H>
        TensorRTHolder<ICudaEngine> createEngine(IBuilder* builder,
            std::function<B*(INetworkDefinition*, std::map<std::string, Weights>&, ITensor&, const FastreidConfig&)> backbone,
            std::function<H*(INetworkDefinition*, std::map<std::string, Weights>&, ITensor&, const FastreidConfig&)> head) {

            auto network = make_holder(builder->createNetworkV2(0U));
            auto config = make_holder(builder->createBuilderConfig());
            auto data = network->addInput(_engineCfg.input_name.c_str(), _dt, Dims3{3, _engineCfg.input_h, _engineCfg.input_w});
            TRTASSERT(data);

            auto weightMap = loadWeights(_engineCfg.weights_path);

            /* Preprocessing */
            auto pre_input = preprocessing_gpu(network.get(), weightMap, data);
            if (!pre_input) pre_input = data;
   
            /* Modeling */
            auto feat_map = backbone(network.get(), weightMap, *pre_input, _reidcfg);
            TRTASSERT(feat_map);
            auto embedding = head(network.get(), weightMap, *feat_map->getOutput(0), _reidcfg);
            TRTASSERT(embedding);

            /* Set output */
            embedding->getOutput(0)->setName(_engineCfg.output_name.c_str());
            network->markOutput(*embedding->getOutput(0));

            /* Build engine */ 
            builder->setMaxBatchSize(_engineCfg.max_batch_size);
            config->setMaxWorkspaceSize(1 << 20);
#ifdef BUILD_FP16
            std::cout << "[Build fp16]" << std::endl;
            config->setFlag(BuilderFlag::kFP16);
#endif 
            auto engine = make_holder(builder->buildEngineWithConfig(*network, *config));
            std::cout << "[TRT engine build out]" << std::endl;

            for (auto& mem : weightMap) {
                free((void*) (mem.second.values));
            }
            return engine;
        }

        virtual void preprocessing_cpu(const cv::Mat& img, float* const data, const std::size_t stride) = 0;
        virtual ITensor* preprocessing_gpu(INetworkDefinition* network, 
            std::map<std::string, Weights>& weightMap, 
            ITensor* input) { return nullptr; };

    private:
        DataType _dt{DataType::kFLOAT};
        trt::EngineConfig _engineCfg;
        FastreidConfig _reidcfg;
        std::unique_ptr<trt::InferenceEngine> _inferEngine{nullptr};
    };
}
