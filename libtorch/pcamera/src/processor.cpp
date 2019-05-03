#include <iostream>
#include <string>
#include <sstream>
#include "processor.h"

Processor::Processor(const char* trace_file, float f) : l{0},st_region{0, 0}, ed_region{0, 0}, threshold{f} {
    this->model = torch::jit::load(trace_file);
    assert(model != nullptr);
}

void Processor::process(cv::Mat& frame) {
    if (this->l == 0 &&
            this->st_region == cv::Point{0, 0} &&
            this->ed_region == cv::Point{0, 0}) {
        this->setRegion(frame);
    }
    cv::Mat data;
    int label;
    float prob;
    bool hit = false;

    this->extractData(frame, data);
    std::tie(label, prob) = this->predict(data);
    if (prob > this->threshold) {
        std::cout << label << ":" << prob << std::endl;
        hit = true;
        this->drawLabel(frame, label, prob);
    }
    this->drawRect(frame, hit);
    data.release();
}

void Processor::setRegion(cv::Mat& frame) {
    int edge = frame.rows < frame.cols ? frame.rows : frame.cols;
    this->l = edge/3;
    this->st_region.x = (frame.cols - this->l)/2;
    this->st_region.y = (frame.rows - this->l)/2;
    this->ed_region.x = (frame.cols + this->l)/2;
    this->ed_region.y = (frame.rows + this->l)/2;
}

void Processor::extractData(cv::Mat& frame, cv::Mat& data) {
    cv::Mat src, gray, invert;
    cv::Mat raw(frame, cv::Rect(this->st_region.x, this->st_region.y, this->l, this->l));
    cv::resize(raw, src, cv::Size(), 28.0f/raw.cols, 28.0f/raw.rows);
    cv::cvtColor(src, gray, cv::COLOR_BGR2GRAY);
    cv::threshold(gray, invert, 0, 255, cv::THRESH_BINARY_INV|cv::THRESH_OTSU);
    invert.convertTo(data, CV_32FC1, 1.0f/255.0f);
    src.release();
    gray.release();
    invert.release();
}

std::tuple<int, float> Processor::predict(cv::Mat& data) {
    auto input = torch::from_blob(data.data, {1, 1, 28, 28}, at::kFloat);
    auto output = at::softmax(this->model->forward({input}).toTensor(), 1);
    auto pred = at::argmax(output, 1, true);
    auto label = pred.item<int>();
    auto prob = output[0][label].item<float_t>();
    return std::forward_as_tuple(label, prob);
}

void Processor::drawRect(cv::Mat& frame, bool hit) {
    auto color = hit ? cv::Scalar(200, 0, 0) : cv::Scalar(0, 200, 0);
    cv::rectangle(frame, this->st_region, this->ed_region, color, 3, 8);
}

void Processor::drawLabel(cv::Mat& frame, int label, float prob) {
    std::stringstream ostr;
    auto color = cv::Scalar(200, 0, 0);
    auto pos = cv::Point(this->st_region.x, this->st_region.y - 20);
    ostr << "predicted: '" << label << "' (probability : " << prob << ")" << std::endl;
    cv::putText(frame, ostr.str(), pos, cv::FONT_HERSHEY_SIMPLEX, 1, color, 2, 8);
}
