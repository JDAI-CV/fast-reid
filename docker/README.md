# Use the container

### Build Container

```shell script
cd docker/
# Build with the corresponding CUDA version

# CUDA 10
docker build -t=fastreid:v0 -f Dockerfile_CUDA10 .
# CUDA 11
docker build -t=fastreid:v0 -f Dockerfile_CUDA11 .
```

### Run Container

```
# Launch (requires GPUs)
nvidia-docker run -v ${PWD}:/home/appuser --name=fastreid --net=host --ipc=host -it fastreid:v0
```

### Run Training

Next, follow the [Get Started Doc](https://github.com/JDAI-CV/fast-reid/blob/master/GETTING_STARTED.md#compile-with-cython-to-accelerate-evalution).


## A more complete docker container

If you want to use a complete docker container which contains many useful tools, you can check my development environment [Dockerfile](https://github.com/L1aoXingyu/fastreid_docker)
