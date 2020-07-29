# encoding: utf-8
"""
@author:  xingyu liao
@contact: sherlockliao01@gmail.com
"""

import argparse
import os
import numpy as np
import sys

sys.path.append('../../')
sys.path.append("/export/home/lxy/runtimelib-tensorrt-tiny/build")

import pytrt
from fastreid.utils.logger import setup_logger
from fastreid.utils.file_io import PathManager


logger = setup_logger(name='trt_export')


def get_parser():
    parser = argparse.ArgumentParser(description="Convert ONNX to TRT model")

    parser.add_argument(
        "--name",
        default="baseline",
        help="name for converted model"
    )
    parser.add_argument(
        "--output",
        default='outputs/trt_model',
        help='path to save converted trt model'
    )
    parser.add_argument(
        "--onnx-model",
        default='outputs/onnx_model/baseline.onnx',
        help='path to onnx model'
    )
    parser.add_argument(
        "--height",
        type=int,
        default=256,
        help="height of image"
    )
    parser.add_argument(
        "--width",
        type=int,
        default=128,
        help="width of image"
    )
    return parser


def export_trt_model(onnxModel, engineFile, input_numpy_array):
    r"""
    Export a model to trt format.
    """

    trt = pytrt.Trt()

    customOutput = []
    maxBatchSize = 1
    calibratorData = []
    mode = 2
    trt.CreateEngine(onnxModel, engineFile, customOutput, maxBatchSize, mode, calibratorData)
    trt.DoInference(input_numpy_array)  # slightly different from c++
    return 0


if __name__ == '__main__':
    args = get_parser().parse_args()

    inputs = np.zeros(shape=(32, args.height, args.width, 3))
    onnxModel = args.onnx_model
    engineFile = os.path.join(args.output, args.name+'.engine')

    PathManager.mkdirs(args.output)
    export_trt_model(onnxModel, engineFile, inputs)

    logger.info(f"Export trt model in {args.output} successfully!")
