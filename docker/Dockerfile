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
WORKDIR /workspace

# Copy the requirements file into the container at /usr/src/app
COPY requirements.txt ./

# Install any needed packages specified in requirements.txt
RUN python3 -m pip install --no-cache-dir -r requirements.txt

# Copy the rest of your application's code into the container
COPY . .
