# Start from the CUDA 11.3 base image
FROM nvidia/cuda:11.1.1-base-ubuntu20.04

# Install Python 3.8 and other essential packages
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    python3.8 \
    python3-pip \
    python3.8-dev \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Set the python3.8 as the default python version
RUN update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.8 1 \
    && update-alternatives --set python3 /usr/bin/python3.8

# Update pip to the latest version
RUN python3 -m pip install --upgrade pip

# Set the working directory in the container
WORKDIR /workspace/src
RUN mkdir /workspace/src/fast-reid

# Copy the requirements file into the container at /workspace/src
COPY . /workspace/src

# Install any needed packages specified in requirements.txt
RUN pip3 install torch==1.8.1+cu111 torchvision==0.9.1+cu111 torchaudio==0.8.1 -f https://download.pytorch.org/whl/torch_stable.html
RUN python3 -m pip install --no-cache-dir -r /workspace/src/fast-reid/docker/requirements.txt

# Build Cython Eval
RUN cd /workspace/src/fast-reid/fastreid/evaluation/rank_cylib && make all
