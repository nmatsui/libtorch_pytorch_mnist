#ifndef __PROCESSOR_H_INCLUDED_
#define __PROCESSOR_H_INCLUDED_

#include <opencv2/opencv.hpp>
#include <torch/torch.h>
#include <torch/script.h>

class Processor {
    int l;
    cv::Point_<int> st_region;
    cv::Point_<int> ed_region;
    float threshold;
    std::shared_ptr<torch::jit::script::Module> model;

    void setRegion(cv::Mat&);
    void extractData(cv::Mat&, cv::Mat&);
    std::tuple<int, float> predict(cv::Mat&);
    void drawRect(cv::Mat&, bool);
    void drawLabel(cv::Mat&, int, float);

public:
    Processor(const char*, float);
    Processor(Processor const&) = default;
    Processor(Processor&&) = default;
    Processor& operator =(Processor const&) = default;
    Processor& operator =(Processor&&) = default;
    ~Processor() = default;

    void process(cv::Mat&);
};

#endif
