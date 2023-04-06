ARG BASE_TORCH_VERSION=pth1.13
ARG BASE_IMAGE=dustynv/l4t-pytorch:r35.1.0-pth1.13-py3

#ARG BASE_IMAGE=nvcr.io/nvidia/l4t-pytorch:r35.1.0-${BASE_TORCH_VERSION}-py3
FROM ${BASE_IMAGE}

ENV DEBIAN_FRONTEND noninteractive

ARG USER_ID
ARG GROUP_ID
ARG USER_NAME
# removing wifi/ip error
RUN apt-get clean
RUN rm -rf /var/atp/lists/*
# this line change to ubuntu archive
RUN apt-get update -y
RUN apt-get upgrade -y
RUN apt-get update \
    && apt-get install -qq -y --no-install-recommends \
    build-essential \
    cmake \
    python3 \
    python3-setuptools \
    python3-pip \
    python3-dev \
    git \
    libssl-dev \
    libusb-1.0-0-dev \
    pkg-config \
    libgtk-3-dev \
    libglfw3-dev \
    libgl1-mesa-dev \
    libglu1-mesa-dev \
    curl \
    ca-certificates
# required pyrealsense2 build (cmake -DPYTHON_EXECUTABLE and PYBINDING)
RUN apt-get install python3.9-dev -y


######################################
#   librealsense Base Image Stage    #
######################################

ARG USER_ID
ARG GROUP_ID
ARG USER_NAME
RUN addgroup --gid $GROUP_ID $USER_NAME --force-badname
RUN adduser --disabled-password --gecos '' --uid $USER_ID --gid $GROUP_ID $USER_NAME --force-badname
RUN usermod -a -G video $USER_NAME

RUN echo $USER_NAME:$USER_NAME|chpasswd
RUN adduser ${USER_NAME} sudo
RUN echo ${USER_NAME}' ALL=NOPASSWD: ALL' >> /etc/sudoers

ENV PYTHONPATH=$PYTHONPATH:/usr/local/lib

# # Install dep packages (for opencv & pyqt)
RUN apt-get update && apt-get upgrade -y \
    && apt-get install -y --no-install-recommends \
    libusb-1.0-0 \
    udev \
    apt-transport-https \
    ca-certificates \
    curl \
    software-properties-common \
    python3-pyqt5*

# visualize app and camera tool updates
RUN apt-get install vim -y
RUN apt-get install tmux -y
RUN apt-get install libgl1-mesa-glx -y
RUN apt-get install ffmpeg -y
RUN apt-get install libx264-dev -y
RUN apt-get install x11-xserver-utils -y
RUN apt-get install x11-apps -y
RUN apt-get install -y libxinerama-dev libxcursor-dev
RUN apt-get install v4l-utils -y
RUN apt-get install '^libxcb.*-dev' libx11-xcb-dev libglu1-mesa-dev libxrender-dev libxi-dev libxkbcommon-dev libxkbcommon-x11-dev -y

RUN apt-get autoremove -y
RUN apt-get clean

# get librealsense
WORKDIR /usr/src
RUN echo $PWD
RUN git clone https://github.com/IntelRealSense/librealsense.git
RUN ls $PWD

# Build and install librealsense
RUN cd /usr/src/librealsense \
    && mkdir build && cd build \
    && cmake ../ \
    -DBUILD_GRAPHICAL_EXAMPLES=OFF \
    -DBUILD_PYTHON_BINDINGS:bool=true \
    -DPYTHON_EXECUTABLE=/usr/bin/python3 \
    -DCMAKE_BUILD_TYPE=release \
    -DFORCE_RSUSB_BACKEND=false
RUN cd /usr/src/librealsense/build && make clean && make -j$(($(nproc)-1)) all && make install

# send library librealsense2 to python3.8
RUN cp /usr/src/librealsense/build/wrappers/python/pyrealsense2.cpython-38-aarch64-linux-gnu.so /usr/local/lib/


# #########################
# #    python setting     #
# #########################
RUN python3 -m pip install --upgrade --no-cache-dir setuptools pip

# RUN python3 -m pip install --no-cache-dir opencv-python
# in Dockerfile, can't use cuda to RUN python3 -m pip install opencv-contrib-python
RUN python3 -m pip install --no-cache-dir --upgrade Cython
RUN python3 -m pip install --no-cache-dir scipy
RUN python3 -m pip install --no-cache-dir tqdm
RUN pip3 install -U easydict
RUN pip3 install -U onnxruntime
RUN pip3 install -U onnx-simplifier==0.3.5
RUN pip3 install -U onnxsim
# if pre-installed opencv, it does not run docker container
# RUN pip3 install -U opencv-python==4.5.5.64

RUN pip3 install -U seaborn
RUN pip3 install -U loguru
RUN pip3 install -U numpy==1.22.1 # same setting trt version
RUN pip3 install -U pyyaml

USER ${USER_NAME}
WORKDIR /home/${USER_NAME}
RUN git clone https://github.com/pdh930105/yolov7-face.git
RUN cd yolov7-face

ENV NVIDIA_VISIBLE_DEVICES all
ENV NVIDIA_DRIVER_CAPABILITIES all
ENV PATH=/usr/local/cuda/bin:/usr/local/cuda/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin
ENV CUDA_HOME=/usr/local/cuda
ENV LD_LIBRARY_PATH=/usr/local/cuda/lib64:/usr/local/cuda/lib64
ENV DEBIAN_FRONTEND=noninteractive
