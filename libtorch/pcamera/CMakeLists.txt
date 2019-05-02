cmake_minimum_required(VERSION 3.1)

project (libtorch_mnist_camera CXX)

add_executable(predict_camera src/predict_camera.cpp src/processor.cpp)
target_link_libraries(predict_camera ${OpenCV_LIBS} ${TORCH_LIBRARIES})
