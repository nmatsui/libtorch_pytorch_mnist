cmake_minimum_required(VERSION 3.1)

project (libtorch_mnist_image CXX)

add_executable(predict_image src/predict_image.cpp)
target_link_libraries(predict_image ${OpenCV_LIBS} ${TORCH_LIBRARIES})
