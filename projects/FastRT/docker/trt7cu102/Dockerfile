# cuda10.2
FROM nvcr.io/nvidia/tensorrt:20.03-py3

ENV DEBIAN_FRONTEND=noninteractive

RUN add-apt-repository -y ppa:timsc/opencv-3.4 && \
apt-get update && \
apt-get install -y cmake \
libopencv-dev \
libopencv-dnn-dev \
libopencv-shape3.4-dbg && \
rm -rf /var/lib/apt/lists/*

