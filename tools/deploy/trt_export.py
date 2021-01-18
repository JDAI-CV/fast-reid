# encoding: utf-8
"""
@author:  xingyu liao
@contact: sherlockliao01@gmail.com
"""

import argparse
import os
import numpy as np
import sys
import tensorrt as trt

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


def onnx2trt(
        model,
        save_path,
        log_level='ERROR',
        max_batch_size=1,
        max_workspace_size=1,
        fp16_mode=False,
        strict_type_constraints=False,
        int8_mode=False,
        int8_calibrator=None,
):
    """build TensorRT model from onnx model.
    Args:
        model (string or io object): onnx model name
        log_level (string, default is ERROR): tensorrt logger level, now
            INTERNAL_ERROR, ERROR, WARNING, INFO, VERBOSE are support.
        max_batch_size (int, default=1): The maximum batch size which can be used at execution time, and also the
            batch size for which the ICudaEngine will be optimized.
        max_workspace_size (int, default is 1): The maximum GPU temporary memory which the ICudaEngine can use at
            execution time. default is 1GB.
        fp16_mode (bool, default is False): Whether or not 16-bit kernels are permitted. During engine build
            fp16 kernels will also be tried when this mode is enabled.
        strict_type_constraints (bool, default is False): When strict type constraints is set, TensorRT will choose
            the type constraints that conforms to type constraints. If the flag is not enabled higher precision
            implementation may be chosen if it results in higher performance.
        int8_mode (bool, default is False): Whether Int8 mode is used.
        int8_calibrator (volksdep.calibrators.base.BaseCalibrator, default is None): calibrator for int8 mode,
            if None, default calibrator will be used as calibration data.
    """

    logger = trt.Logger(getattr(trt.Logger, log_level))
    builder = trt.Builder(logger)

    network = builder.create_network(1 << (int)(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
    parser = trt.OnnxParser(network, logger)
    if isinstance(model, str):
        with open(model, 'rb') as f:
            flag = parser.parse(f.read())
    else:
        flag = parser.parse(model.read())
    if not flag:
        for error in range(parser.num_errors):
            print(parser.get_error(error))

    # re-order output tensor
    output_tensors = [network.get_output(i) for i in range(network.num_outputs)]
    [network.unmark_output(tensor) for tensor in output_tensors]
    for tensor in output_tensors:
        identity_out_tensor = network.add_identity(tensor).get_output(0)
        identity_out_tensor.name = 'identity_{}'.format(tensor.name)
        network.mark_output(tensor=identity_out_tensor)

    builder.max_batch_size = max_batch_size

    config = builder.create_builder_config()
    config.max_workspace_size = max_workspace_size * (1 << 25)
    if fp16_mode:
        config.set_flag(trt.BuilderFlag.FP16)
    if strict_type_constraints:
        config.set_flag(trt.BuilderFlag.STRICT_TYPES)
    # if int8_mode:
    #     config.set_flag(trt.BuilderFlag.INT8)
    #     if int8_calibrator is None:
    #         shapes = [(1,) + network.get_input(i).shape[1:] for i in range(network.num_inputs)]
    #         dummy_data = utils.gen_ones_data(shapes)
    #         int8_calibrator = EntropyCalibrator2(CustomDataset(dummy_data))
    #     config.int8_calibrator = int8_calibrator

    # set dynamic batch size profile
    profile = builder.create_optimization_profile()
    for i in range(network.num_inputs):
        tensor = network.get_input(i)
        name = tensor.name
        shape = tensor.shape[1:]
        min_shape = (1,) + shape
        opt_shape = ((1 + max_batch_size) // 2,) + shape
        max_shape = (max_batch_size,) + shape
        profile.set_shape(name, min_shape, opt_shape, max_shape)
    config.add_optimization_profile(profile)

    engine = builder.build_engine(network, config)

    with open(save_path, 'wb') as f:
        f.write(engine.serialize())
    # trt_model = TRTModel(engine)

    # return trt_model


def export_trt_model(onnxModel, engineFile, input_numpy_array):
    r"""
    Export a model to trt format.
    """

    trt = pytrt.Trt()

    customOutput = []
    maxBatchSize = 8
    calibratorData = []
    mode = 0
    trt.CreateEngine(onnxModel, engineFile, customOutput, maxBatchSize, mode, calibratorData)
    trt.DoInference(input_numpy_array)  # slightly different from c++
    return 0


if __name__ == '__main__':
    args = get_parser().parse_args()

    inputs = np.zeros(shape=(1, args.height, args.width, 3))
    onnxModel = args.onnx_model
    engineFile = os.path.join(args.output, args.name+'.engine')

    PathManager.mkdirs(args.output)
    onnx2trt(onnxModel, engineFile)
    # export_trt_model(onnxModel, engineFile, inputs)

    logger.info(f"Export trt model in {args.output} successfully!")
