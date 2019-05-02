#include <iostream>
#include <opencv2/opencv.hpp>
#include <torch/torch.h>
#include <torch/script.h>

void loadImage(const char *fileName, cv::Mat& data) {
    cv::Mat src, gray, invert;
    auto raw = cv::imread(cv::String(fileName));
    assert(!raw.empty());

    cv::resize(raw, src, cv::Size(), 28.0f/raw.cols, 28.0f/raw.rows);
    cv::cvtColor(src, gray, cv::COLOR_BGR2GRAY);
    cv::bitwise_not(gray, invert);
    invert.convertTo(data, CV_32FC1, 1.0f/255.0f);
    src.release();
    gray.release();
    invert.release();
}

int main(int argc, char* argv[]) {
    if (argc != 3) {
        std::cout << "usage: " << argv[0] << " trace_file image_file" << std::endl;
        return 1;
    }

    cv::Mat data;
    loadImage(argv[2], data);
    auto input = torch::from_blob(data.data, {1, 1, 28, 28}, at::kFloat);

    auto model = torch::jit::load(argv[1]);
    assert(model != nullptr);

    auto output = at::softmax(model->forward({input}).toTensor(), 1);
    auto pred = at::argmax(output, 1, true);
    auto label = pred.item<int64_t>();
    auto prob = output[0][label].item<float_t>();
    std::cout << "label: " << label << " (prob: " << prob << ")" << std::endl;
    data.release();

    return 0;
}
