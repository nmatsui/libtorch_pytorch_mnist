#include <iostream>
#include <string>
#include <opencv2/opencv.hpp>
#include "processor.h"

const static int frame_rate = 12;

void captureFrame(Processor& processor, int device_no) {
    cv::VideoCapture cap(device_no);
    if (!cap.isOpened()) {
        std::cout << "cannot open device (" << device_no << ")" << std::endl;
        assert(false);
    }
    cap.set(cv::CAP_PROP_FPS, frame_rate);
    std::cout << cap.get(cv::CAP_PROP_FPS) << std::endl;

    cv::Mat frame;
    while(cap.read(frame)) {
        processor.process(frame);
        cv::imshow("win", frame);
        const int key = cv::waitKey(1);
        if(key == 'q') {
            break;
        }
    }
    cap.release();
    cv::destroyAllWindows();
}

int main(int argc, char* argv[]) {
    if (argc != 4) {
        std::cout << "usage: " << argv[0] << " trace_file device_no threshold" << std::endl;
        return 1;
    }

    int device_no;
    float threshold;
    try {
        device_no = std::stoi(argv[2]);
        threshold = std::stof(argv[3]);
    }
    catch (...) {
        std::cout << "device_no (" << argv[2] << ") or threshold (" << argv[3] << ") is invalid" << std::endl;
        assert(false);
    }

    auto processor = Processor(argv[1], threshold);
    captureFrame(processor, device_no);

    return 0;
}
