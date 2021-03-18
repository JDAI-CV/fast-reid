# cuda10.0
FROM fineyu/tensorrt7:0.0.1

ENV DEBIAN_FRONTEND=noninteractive

RUN add-apt-repository -y ppa:timsc/opencv-3.4 && \
apt-get update && \
apt-get install -y cmake \
libopencv-dev \
libopencv-dnn-dev \
libopencv-shape3.4-dbg && \
rm -rf /var/lib/apt/lists/*
