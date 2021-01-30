#include "fastrt/model.h"

namespace fastrt {

    Model::Model(const trt::ModelConfig &modelcfg, const FastreidConfig &reidcfg, const std::string input_name, const std::string output_name) :
        _reidcfg(reidcfg) {
        
        _engineCfg.weights_path = modelcfg.weights_path;
        _engineCfg.max_batch_size = modelcfg.max_batch_size;
        _engineCfg.input_h = modelcfg.input_h;
        _engineCfg.input_w = modelcfg.input_w;
        _engineCfg.output_size = modelcfg.output_size;
        _engineCfg.device_id = modelcfg.device_id;

        _engineCfg.input_name = input_name;
        _engineCfg.output_name = output_name;       
        _engineCfg.trtModelStream = nullptr;
        _engineCfg.stream_size = 0;
    };

    bool Model::deserializeEngine(const std::string engine_file) {
        std::ifstream file(engine_file, std::ios::binary | std::ios::in);
        if (file.good()) {
            file.seekg(0, file.end);
            _engineCfg.stream_size = file.tellg();
            file.seekg(0, file.beg);
            _engineCfg.trtModelStream = std::shared_ptr<char>( new char[_engineCfg.stream_size], []( char* ptr ){ delete [] ptr; } );
            TRTASSERT(_engineCfg.trtModelStream.get());
            file.read(_engineCfg.trtModelStream.get(), _engineCfg.stream_size);
            file.close();
    
            _inferEngine = make_unique<trt::InferenceEngine>(_engineCfg);
            return true;
        }
        return false;
    }

    bool Model::inference(std::vector<cv::Mat> &input) {
        if (_inferEngine != nullptr) {
            const std::size_t stride = _engineCfg.input_h * _engineCfg.input_w;
            return _inferEngine.get()->doInference(input.size(), 
                [&](float* data) {
                    for(const auto &img : input) {
                        preprocessing_cpu(img, data, stride);
                        data += 3 * stride;
                    }
                }
            );
        } else {
            return false;
        }
    }

    float* Model::getOutput() { 
        if(_inferEngine != nullptr) 
            return _inferEngine.get()->getOutput(); 
        return nullptr;
    }

    int Model::getOutputSize() { 
        return _engineCfg.output_size; 
    }

    int Model::getDeviceID() { 
        return _engineCfg.device_id; 
    }
}
