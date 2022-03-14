# How to run
## Without option to add your own images
```
cd paintings-animator
open index.html
```

## With option to add your own images
Remove `opencv` from `requirements.txt` and build it from source if you would like to use CUDA.
```
cd paintings-animator
pip install -r requirements.txt
python src/main.py
open localhost:5000
```

# CUDA
This section is needed only if you would like to use CUDA for transformation of images.  


Install cuda toolkit v11.5
```
sudo apt-key adv --fetch-keys http://developer.download.nvidia.com/compute/cuda/repos/ubuntu1804/x86_64/7fa2af80.pub
sudo sh -c 'echo "deb http://developer.download.nvidia.com/compute/cuda/repos/ubuntu1804/x86_64 /" > /etc/apt/sources.list.d/cuda.list'
sudo apt-get update
sudo apt-get install -y cuda-toolkit-11-5
```
Check cuda devices
```
cd /usr/local/cuda/samples/1_Utilities/deviceQuery
sudo make
./deviceQuery
```
Your GPU should be present in the output.

## opencv
Opencv needs to be built with cuda enabled.  

First clone the source code:
```
git clone -b 4.5.5 https://github.com/opencv/opencv.git
git clone -b piecewiseAffineCuda https://github.com/jjustin/opencv_contrib.git

cd opencv 
mkdir build
cd build
```
Configure enviromental variables to your build. `CUDA_ARCH_BIN` can be found [here](https://developer.nvidia.com/cuda-gpus).
Call `cmake`:
```
export PYTHON_VERSION=3.8.12

cmake -D CMAKE_BUILD_TYPE=RELEASE \
-D CMAKE_INSTALL_PREFIX=/usr/local \
-D OPENCV_ENABLE_NONFREE=ON \
-D WITH_CUDA=ON \
-D WITH_CUDNN=ON \
-D OPENCV_DNN_CUDA=ON \
-D ENABLE_FAST_MATH=1 \
-D CUDA_FAST_MATH=1 \
-D CUDA_ARCH_BIN=8.6 \
-D WITH_CUBLAS=1 \
-D OPENCV_EXTRA_MODULES_PATH=$(pwd)/../../opencv_contrib/modules ..

# Optional additional parameters for building for pyenv, change `python3.8` to desired interpreter/version
-D PYTHON3_EXECUTABLE=$HOME/.pyenv/versions/$PYTHON_VERSION/bin/python3.8 \
-D PYTHON3_PACKAGES_PATH=$HOME/.pyenv/versions/$PYTHON_VERSION/lib/python3.8/site-packages \
-D PYTHON3_LIBRARY=$HOME/.pyenv/versions/$PYTHON_VERSION/lib/libpython3.8.a \
-D PYTHON3_INCLUDE_DIR=$HOME/.pyenv/versions/$PYTHON_VERSION/include \
```
Ensure correct python environment is present in the output. And check CUDA is being built.  

Build and install using:
```
make -j$(nproc)
sudo make install
```


[See also](https://cuda-chen.github.io/image%20processing/programming/2020/02/22/build-opencv-dnn-module-with-nvidia-gpu-support-on-ubuntu-1804.html)