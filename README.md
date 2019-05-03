# pytorch mnist
You can train your MNIST CNN model by using pytorch (python3), and you can predict a handwritten digit by using libtorch (C++11).

## Requirements
* Ubuntu 18.04.2 LTS
* miniconda3-4.3.30
* python 3.7
* opencv 4.1.0
* pytorch 1.1.0

## prepare
### install required libraries
1. install libraries

    ```
    $ sudo apt install -y build-essential cmake unzip pkg-config wget
    $ sudo apt install -y qt5-default libvtk6-dev zlib1g-dev libwebp-dev libjasper-dev \
                          libopenexr-dev libgdal-dev libjpeg-dev libpng-dev libtiff-dev libtiff5-dev \
                          libavcodec-dev libavformat-dev libswscale-dev libv4l-dev libxine2-dev \
                          libxvidcore-dev libx264-dev libdc1394-22-dev libtheora-dev libvorbis-dev \
                          libgtk-3-dev libatlas-base-dev libopencore-amrnb-dev  libopencore-amrwb-dev \
                          libtbb-dev libeigen3-dev gfortran yasm
    ```
1. install opencv

    ```
    $ cd ${HOME}
    $ wget -O opencv-4.1.0.zip https://github.com/opencv/opencv/archive/4.1.0.zip
    $ unzip opencv-4.1.0.zip
    $ cd opencv-4.1.0
    $ mkdir build && cd build
    $ cmake -DCMAKE_BUILD_TYPE=RELEASE \
            -DCMAKE_INSTALL_PREFIX=/usr/local \
            -DCMAKE_CXX_FLAGS=-D_GLIBCXX_USE_CXX11_ABI=0 \
            -DWITH_QT=ON -DWITH_OPENGL=ON -DFORCE_VTK=ON -DWITH_TBB=ON -DWITH_GDAL=ON \
            -DWITH_XINE=ON -DBUILD_EXAMPLES=ON -DENABLE_PRECOMPILED_HEADERS=OFF \
            ..
    $ make -j4
    $ sudo make install
    $ sudo ldconfig
    ```
1. install libtorch

    ```
    $ cd ${HOME}
    $ wget -O libtorch-1.1.zip https://download.pytorch.org/libtorch/cpu/libtorch-shared-with-deps-latest.zip
    $ unzip libtorch-1.1.zip
    $ sudo cp -r libtorch/include/* /usr/local/include/
    $ sudo cp -r libtorch/lib/* /usr/local/lib/
    $ sudo cp -r libtorch/share/* /usr/local/share/
    $ sudo ldconfig
    ```

### build C++ programs
1. cd `libtorch` directory

    ```
    $ cd ${HOME}/pytorch_mnist/libtorch
    ```
1. prepare build directory

    ```
    $ mkdir build && cd build
    ```
1. build

    ```
    $ cmake ..
    $ make
    $ cp pimage/predict_image ..
    $ cp pcamera/predict_camera ..
    ```

## How to train (python)
1. go to the python source directory

    ```
    $ cd ${HOME}/pytorch_mnist/pytorch
    ```
1. create virtualenv and install required package by using conda

    ```
    $ conda env create --file conda.yaml
    $ conda activate pytorch_mnist
    ```
1. train data

    ```
    $ ./train.py --epochs 12 ../models/mnist_py.pt ../data
    ```
    * 1st argument: the model weights file to be trained
    * 2nd argument: the root directory to be saved MNIST data
    * when training is complete, the loss and accuracy will be displayed like below:

        ```
        Test set: Average loss: 0.050700, Accuracy: 9847/10000 (98.47%)
        ```
1. convert the model weights file to be able to use by c++

    ```
    $ ./convert_model.py ../models/mnist_py.pt ../models/mnist_cpp.pt
    ```
    * 1st argument: trained model weights file
    * 2nd argument: the model weights file to be converted for c++

## How to predict a handwritten digit from a image file (c++)
1. go to the c++ source directory

    ```
    $ cd ${HOME}/pytorch_mnist/libtorch
    ```
1. predict a handwitten digit like below:

    ```
    $ ./predict_image ../models/mnist_cpp.pt ../digit_images/5.png
    ```
    * 1st argument: converted model weights file for c++
    * 2nd argument: a handwritten digit image file

## How to predict a handwritten digit continuously from USB camera frame (c++)
1. go to the c++ source directory

    ```
    $ cd ${HOME}/pytorch_mnist/libtorch
    ```

## How to predict a handwritten digit from a image file (python)
1. go to the python source directory

    ```
    $ cd ${HOME}/pytorch_mnist/pytorch
    ```
1. predict a handwitten digit like below:

    ```
    $ ./predict.py ../models/mnist_py.pt ../digit_images/5.png 
    ```
    * 1st argument: trained model weights file
    * 2nd argument: a handwritten digit image file

## How to predict a handwritten digit from a image file (c++)
1. go to the c++ source directory

    ```
    $ cd ${HOME}/pytorch_mnist/libtorch
    ```
1. predict a handwitten digit like below:

    ```
    $ ./predict_image ../models/mnist_cpp.pt ../digit_images/5.png
    ```
    * 1st argument: converted model weights file for c++
    * 2nd argument: a handwritten digit image file

## How to predict a handwritten digit continuously from USB camera frame (c++)
1. go to the c++ source directory

    ```
    $ cd ${HOME}/pytorch_mnist/libtorch
    ```
1. start camera preview like below:

    ```
    $ ./predict_camera ../models/mnist_cpp.pt 0 0.9

    ```
    * 1st argument: converted model weights file for c++
    * 2nd argument: camera device id
    * 3rd argument: predict the handwritten digit when the calcurated probability is greater than this float
1. predict the digit when a character is displayed in the green box

