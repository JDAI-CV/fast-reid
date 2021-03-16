#include <iostream>
#include <opencv2/opencv.hpp>
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>

#include "fastrt/utils.h"
#include "fastrt/baseline.h"
#include "fastrt/factory.h"
using namespace fastrt;
using namespace nvinfer1;

namespace py = pybind11;

/* Ex1. sbs_R50-ibn */
// static const std::string WEIGHTS_PATH = "../sbs_R50-ibn.wts"; 
// static const std::string ENGINE_PATH = "./sbs_R50-ibn.engine";

// static const int MAX_BATCH_SIZE = 4;
// static const int INPUT_H = 256;
// static const int INPUT_W = 128;
// static const int OUTPUT_SIZE = 2048;
// static const int DEVICE_ID = 0;

// static const FastreidBackboneType BACKBONE = FastreidBackboneType::r50; 
// static const FastreidHeadType HEAD = FastreidHeadType::EmbeddingHead;
// static const FastreidPoolingType HEAD_POOLING = FastreidPoolingType::gempoolP;
// static const int LAST_STRIDE = 1;
// static const bool WITH_IBNA = true; 
// static const bool WITH_NL = true;
// static const int EMBEDDING_DIM = 0; 

/* Ex4.kd-r34-r101_ibn */
// static const std::string WEIGHTS_PATH = "../kd-r34-r101_ibn.wts"; 
// static const std::string ENGINE_PATH = "./build/kd_r34_distill.engine"; 

// static const int MAX_BATCH_SIZE = 16;
// static const int INPUT_H = 384;
// static const int INPUT_W = 128;
// static const int OUTPUT_SIZE = 512;
// static const int DEVICE_ID = 0;

// static const FastreidBackboneType BACKBONE = FastreidBackboneType::r34_distill; 
// static const FastreidHeadType HEAD = FastreidHeadType::EmbeddingHead;
// static const FastreidPoolingType HEAD_POOLING = FastreidPoolingType::gempoolP;
// static const int LAST_STRIDE = 1;
// static const bool WITH_IBNA = true; 
// static const bool WITH_NL = false;
// static const int EMBEDDING_DIM = 0; 


/* Ex5.kd-r18-r101_ibn */
static const std::string WEIGHTS_PATH = "../kd-r18-r101_ibn.wts"; 
static const std::string ENGINE_PATH = "./kd_r18_distill.engine"; 

static const int MAX_BATCH_SIZE = 16;
static const int INPUT_H = 384;
static const int INPUT_W = 128;
static const int OUTPUT_SIZE = 512;
static const int DEVICE_ID = 1;

static const FastreidBackboneType BACKBONE = FastreidBackboneType::r18_distill; 
static const FastreidHeadType HEAD = FastreidHeadType::EmbeddingHead;
static const FastreidPoolingType HEAD_POOLING = FastreidPoolingType::gempoolP;
static const int LAST_STRIDE = 1;
static const bool WITH_IBNA = true; 
static const bool WITH_NL = false;
static const int EMBEDDING_DIM = 0; 

FastreidConfig reidCfg { 
        BACKBONE,
        HEAD,
        HEAD_POOLING,
        LAST_STRIDE,
        WITH_IBNA,
        WITH_NL,
        EMBEDDING_DIM};

class ReID
{

private:
    int device;  // GPU id
    fastrt::Baseline baseline;

public:
    ReID(int a);
    int build(const std::string &engine_file);
    // std::list<float> infer_test(const std::string &image_file);
    std::list<float> infer(py::array_t<uint8_t>&);
    std::list<std::list<float>> batch_infer(py::array_t<uint8_t>&, std::vector<int> &);
    std::list<float> infer_test(const std::string &file_path);
    ~ReID();
};

ReID::ReID(int device): baseline(trt::ModelConfig { 
        WEIGHTS_PATH,
        MAX_BATCH_SIZE,
        INPUT_H,
        INPUT_W,
        OUTPUT_SIZE,
        device})
{
    std::cout << "Init on device " << device << std::endl;
}

int ReID::build(const std::string &engine_file)
{
    if(!baseline.deserializeEngine(engine_file)) {
        std::cout << "DeserializeEngine Failed." << std::endl;
        return -1;
    }
    return 0;
}

ReID::~ReID()
{

    std::cout << "Destroy engine succeed" << std::endl;
}

std::list<float> ReID::infer(py::array_t<uint8_t>& img)
{
    auto rows = img.shape(0);
    auto cols = img.shape(1);
    auto type = CV_8UC3;

    cv::Mat img2(rows, cols, type, (unsigned char*)img.data());
    cv::Mat re(INPUT_H, INPUT_W, CV_8UC3);
    // std::cout << (int)img2.data[0] << std::endl;
    cv::resize(img2, re, re.size(), 0, 0, cv::INTER_CUBIC); /* cv::INTER_LINEAR */
    std::vector<cv::Mat> input;
    input.emplace_back(re);

    if(!baseline.inference(input)) {
        std::cout << "Inference Failed." << std::endl;
    }
    std::list<float> feature;

    float* feat_embedding = baseline.getOutput();
    TRTASSERT(feat_embedding);
    for (int dim = 0; dim < baseline.getOutputSize(); ++dim) {
        feature.push_back(feat_embedding[dim]);
    }

    return feature;
}



std::list<std::list<float>> ReID::batch_infer(py::array_t<uint8_t>& img, std::vector<int> &boxes_array)
{
    // parse to cvmat
    auto rows = img.shape(0);
    auto cols = img.shape(1);
    auto type = CV_8UC3;

    int box_count = boxes_array.size() / 4;

    cv::Mat img2(rows, cols, type, (unsigned char*)img.data());
    std::cout << rows << "," << cols << std::endl;
    std::vector<cv::Mat> input;

    //preprocess
    for (int index = 0; index < box_count; index++)
    {
        cv::Rect myROI(boxes_array.at(index * 4 + 0), boxes_array.at(index * 4 + 1), 
            boxes_array.at(index * 4 + 2) - boxes_array.at(index * 4 + 0), 
            boxes_array.at(index * 4 + 3) - boxes_array.at(index * 4 + 1));
        cv::Mat croppedImage = img2(myROI);
        cv::Mat re(INPUT_H, INPUT_W, CV_8UC3);
        cv::resize(croppedImage, re, re.size(), 0, 0, cv::INTER_CUBIC);
        // std::cout << (int)croppedImage.data[0] << std::endl;
        input.emplace_back(re);
    }
    if(!baseline.inference(input)) {
        std::cout << "Inference Failed." << std::endl;
    }
    std::list<std::list<float>> result;

    float* feat_embedding = baseline.getOutput();
    TRTASSERT(feat_embedding);
    for (int index = 0; index < box_count; index++)
    {
        std::list<float> feature;
        for (int dim = 0; dim < baseline.getOutputSize(); ++dim) {
            feature.push_back(feat_embedding[index * 512 + dim]);
        }
        result.push_back(feature);
    }
    return result;
}




PYBIND11_MODULE(ReID, m) {
    m.doc() = R"pbdoc(
        Pybind11 example plugin
    )pbdoc";
    py::class_<ReID>(m, "ReID")
        .def(py::init<int>())
        .def("build", &ReID::build)
        .def("infer", &ReID::infer, py::return_value_policy::automatic)
        .def("batch_infer", &ReID::batch_infer, py::return_value_policy::automatic)
        ;

#ifdef VERSION_INFO
    m.attr("__version__") = VERSION_INFO;
#else
    m.attr("__version__") = "dev";
#endif
}
