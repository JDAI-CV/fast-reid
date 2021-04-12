# cuda10.0
FROM fineyu/tensorrt7:0.0.1

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y software-properties-common

RUN add-apt-repository -y ppa:timsc/opencv-3.4 && \
apt-get update && \
apt-get install -y cmake \
libopencv-dev \
libopencv-dnn-dev \
libopencv-shape3.4-dbg && \
rm -rf /var/lib/apt/lists/*

RUN curl https://bootstrap.pypa.io/get-pip.py -o get-pip.py && python3 get-pip.py && rm get-pip.py && pip3 install torch==1.6.0 torchvision tensorboard matplotlib scipy Pillow numpy prettytable easydict opencv-python \
scikit-learn pyyaml yacs termcolor tabulate tensorboard opencv-python pyyaml yacs termcolor tabulate gdown faiss-cpu
