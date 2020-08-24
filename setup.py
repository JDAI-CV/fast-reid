from setuptools import find_packages, setup

setup(
    name="fast_reid",
    version="0.1",
    author="JDAI-CV",
    url="https://github.com/JDAI-CV/fast-reid",
    packages=find_packages(exclude=("docs", "tests")),
    python_requires=">=3.6",
    install_requries=[
        "pytorch>=1.6",
        "torchvision",
        "yacs",
        "Cython",
        "tensorboard",
        "gdown",
        "sklearn",
        "termcolor",
        "tabulate",
        "faiss",
    ]

)
