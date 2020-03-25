# encoding: utf-8
"""
@author:  liaoxingyu
@contact: sherlockliao01@gmail.com
"""

import warnings

warnings.filterwarnings('ignore')  # Ignore all the warning messages in this tutorial
from onnx_tf.backend import prepare

import tensorflow as tf
from PIL import Image
import torchvision.transforms as transforms

import onnx
import numpy as np
import torch
from torch.backends import cudnn
import io

cudnn.benchmark = True


def _export_via_onnx(model, inputs):
    from ipdb import set_trace;
    set_trace()

    def _check_val(module):
        assert not module.training

    model.apply(_check_val)

    # Export the model to ONNX
    with torch.no_grad():
        with io.BytesIO() as f:
            torch.onnx.export(
                model,
                inputs,
                f,
                # verbose=True,  # NOTE: uncomment this for debugging
                export_params=True,
            )
            onnx_model = onnx.load_from_string(f.getvalue())
    # torch.onnx.export(model,  # model being run
    #                   inputs,  # model input (or a tuple for multiple inputs)
    #                   "reid_test.onnx",  # where to save the model (can be a file or file-like object)
    # export_params=True,  # store the trained parameter weights inside the model file
    # opset_version=10,  # the ONNX version to export the model to
    # do_constant_folding=True,  # whether to execute constant folding for optimization
    # input_names=['input'],  # the model's input names
    # output_names=['output'],  # the model's output names
    # dynamic_axes={'input': {0: 'batch_size'},  # variable lenght axes
    #               'output': {0: 'batch_size'}})
    # )

    # Apply ONNX's Optimization
    # all_passes = optimizer.get_available_passes()
    # passes = ["fuse_bn_into_conv"]
    # assert all(p in all_passes for p in passes)
    # onnx_model = optimizer.optimize(onnx_model, passes)

    # Convert ONNX Model to Tensorflow Model
    tf_rep = prepare(onnx_model, strict=False)  # Import the ONNX model to Tensorflow
    print(tf_rep.inputs)  # Input nodes to the model
    print('-----')
    print(tf_rep.outputs)  # Output nodes from the model
    print('-----')
    # print(tf_rep.tensor_dict)  # All nodes in the model
    # """

    # install onnx-tensorflow from github，and tf_rep = prepare(onnx_model, strict=False)
    # Reference https://github.com/onnx/onnx-tensorflow/issues/167
    # tf_rep = prepare(onnx_model) # whthout strict=False leads to KeyError: 'pyfunc_0'

    # debug, here using the same input to check onnx and tf.
    # output_onnx_tf = tf_rep.run(to_numpy(img))
    # print('output_onnx_tf = {}'.format(output_onnx_tf))
    # onnx --> tf.graph.pb
    # tf_pb_path = 'reid_tf_graph.pb'
    # tf_rep.export_graph(tf_pb_path)

    return tf_rep


def to_numpy(tensor):
    return tensor.detach().cpu().numpy() if tensor.requires_grad else tensor.cpu().numpy()


def _check_pytorch_tf_model(model: torch.nn.Module, tf_graph_path: str):
    img = Image.open("demo_imgs/dog.jpg")

    resize = transforms.Resize([384, 128])
    img = resize(img)

    to_tensor = transforms.ToTensor()
    img = to_tensor(img)
    img.unsqueeze_(0)
    torch_outs = model(img)

    with tf.Graph().as_default():
        graph_def = tf.GraphDef()
        with open(tf_graph_path, "rb") as f:
            graph_def.ParseFromString(f.read())
            tf.import_graph_def(graph_def, name="")
        with tf.Session() as sess:
            # init = tf.initialize_all_variables()
            # init = tf.global_variables_initializer()
            # sess.run(init)

            # print all ops, check input/output tensor name.
            # uncomment it if you donnot know io tensor names.
            '''
            print('-------------ops---------------------')
            op = sess.graph.get_operations()
            for m in op:
                try:
                    # if 'input' in m.values()[0].name:
                    #     print(m.values())
                    if m.values()[0].shape.as_list()[1] == 2048: #and (len(m.values()[0].shape.as_list()) == 4):
                        print(m.values())
                except:
                    pass
            print('-------------ops done.---------------------')
            '''
            input_x = sess.graph.get_tensor_by_name('input.1:0')  # input
            outputs = sess.graph.get_tensor_by_name('502:0')  # 5
            output_tf_pb = sess.run(outputs, feed_dict={input_x: to_numpy(img)})

    print('output_pytorch = {}'.format(to_numpy(torch_outs)))
    print('output_tf_pb = {}'.format(output_tf_pb))

    np.testing.assert_allclose(to_numpy(torch_outs), output_tf_pb, rtol=1e-03, atol=1e-05)
    print("Exported model has been tested with tensorflow runtime, and the result looks good!")


def export_tf_reid_model(model: torch.nn.Module, tensor_inputs: torch.Tensor, graph_save_path: str):
    """
    Export a reid model via ONNX.
    Arg:
        model: a tf_1.x-compatible version of detectron2 model, defined in caffe2_modeling.py
        tensor_inputs: a list of tensors that caffe2 model takes as input.
    """
    # model = copy.deepcopy(model)
    assert isinstance(model, torch.nn.Module)

    # Export via ONNX
    print("Exporting a {} model via ONNX ...".format(type(model).__name__))
    predict_net = _export_via_onnx(model, tensor_inputs)
    print("ONNX export Done.")

    print("Saving graph of ONNX exported model to {} ...".format(graph_save_path))
    predict_net.export_graph(graph_save_path)

    print("Checking if tf.pb is right")
    _check_pytorch_tf_model(model, graph_save_path)

# if __name__ == '__main__':
# args = default_argument_parser().parse_args()
# print("Command Line Args:", args)
# cfg = setup(args)
# cfg = cfg.defrost()
# cfg.MODEL.BACKBONE.NAME = "build_resnet_backbone"
# cfg.MODEL.BACKBONE.DEPTH = 50
# cfg.MODEL.BACKBONE.LAST_STRIDE = 1
# # If use IBN block in backbone
# cfg.MODEL.BACKBONE.WITH_IBN = True
#
# model = build_model(cfg)
# # model.load_params_wo_fc(torch.load('logs/bjstation/res50_baseline_v0.4/ckpts/model_epoch80.pth'))
# model.cuda()
# model.eval()
# dummy_inputs = torch.randn(1, 3, 256, 128)
# export_tf_reid_model(model, dummy_inputs, 'reid_tf.pb')

# inputs = torch.rand(1, 3, 384, 128).cuda()
#
# _export_via_onnx(model, inputs)
# onnx_model = onnx.load("reid_test.onnx")
# onnx.checker.check_model(onnx_model)
#
# from PIL import Image
# import torchvision.transforms as transforms
#
# img = Image.open("demo_imgs/dog.jpg")
#
# resize = transforms.Resize([384, 128])
# img = resize(img)
#
# to_tensor = transforms.ToTensor()
# img = to_tensor(img)
# img.unsqueeze_(0)
# img = img.cuda()
#
# with torch.no_grad():
#     torch_out = model(img)
#
# ort_session = onnxruntime.InferenceSession("reid_test.onnx")
#
# # compute ONNX Runtime output prediction
# ort_inputs = {ort_session.get_inputs()[0].name: to_numpy(img)}
# ort_outs = ort_session.run(None, ort_inputs)
# img_out_y = ort_outs[0]
#
#
# # compare ONNX Runtime and PyTorch results
# np.testing.assert_allclose(to_numpy(torch_out), ort_outs[0], rtol=1e-03, atol=1e-05)
#
# print("Exported model has been tested with ONNXRuntime, and the result looks good!")

# img = Image.open("demo_imgs/dog.jpg")
#
# resize = transforms.Resize([384, 128])
# img = resize(img)
#
# to_tensor = transforms.ToTensor()
# img = to_tensor(img)
# img.unsqueeze_(0)
# img = torch.cat([img.clone(), img.clone()], dim=0)

# ort_session = onnxruntime.InferenceSession("reid_test.onnx")

# # compute ONNX Runtime output prediction
# ort_inputs = {ort_session.get_inputs()[0].name: to_numpy(img)}
# ort_outs = ort_session.run(None, ort_inputs)

# model = onnx.load('reid_test.onnx')  # Load the ONNX file
# tf_rep = prepare(model, strict=False) # Import the ONNX model to Tensorflow
# print(tf_rep.inputs)  # Input nodes to the model
# print('-----')
# print(tf_rep.outputs)  # Output nodes from the model
# print('-----')
# # print(tf_rep.tensor_dict)  # All nodes in the model

# install onnx-tensorflow from github，and tf_rep = prepare(onnx_model, strict=False)
# Reference https://github.com/onnx/onnx-tensorflow/issues/167
# tf_rep = prepare(onnx_model) # whthout strict=False leads to KeyError: 'pyfunc_0'

# # debug, here using the same input to check onnx and tf.
# # output_onnx_tf = tf_rep.run(to_numpy(img))
# # print('output_onnx_tf = {}'.format(output_onnx_tf))
# # onnx --> tf.graph.pb
# tf_pb_path = 'reid_tf_graph.pb'
# tf_rep.export_graph(tf_pb_path)

# # step 3, check if tf.pb is right.
# with tf.Graph().as_default():
#     graph_def = tf.GraphDef()
#     with open(tf_pb_path, "rb") as f:
#         graph_def.ParseFromString(f.read())
#         tf.import_graph_def(graph_def, name="")
#     with tf.Session() as sess:
#         # init = tf.initialize_all_variables()
#         init = tf.global_variables_initializer()
#         # sess.run(init)

#         # print all ops, check input/output tensor name.
#         # uncomment it if you donnot know io tensor names.
#         '''
#         print('-------------ops---------------------')
#         op = sess.graph.get_operations()
#         for m in op:
#             try:
#                 # if 'input' in m.values()[0].name:
#                 #     print(m.values())
#                 if m.values()[0].shape.as_list()[1] == 2048: #and (len(m.values()[0].shape.as_list()) == 4):
#                     print(m.values())
#             except:
#                 pass
#         print('-------------ops done.---------------------')
#         '''
#         input_x = sess.graph.get_tensor_by_name('input.1:0')  # input
#         outputs = sess.graph.get_tensor_by_name('502:0')  # 5
#         output_tf_pb = sess.run(outputs, feed_dict={input_x: to_numpy(img)})
#         print('output_tf_pb = {}'.format(output_tf_pb))
# np.testing.assert_allclose(ort_outs[0], output_tf_pb, rtol=1e-03, atol=1e-05)

# with tf.Graph().as_default():
#     graph_def = tf.GraphDef()
#     with open(tf_pb_path, "rb") as f:
#         graph_def.ParseFromString(f.read())
#         tf.import_graph_def(graph_def, name="")
#     with tf.Session() as sess:
#         # init = tf.initialize_all_variables()
#         init = tf.global_variables_initializer()
#         # sess.run(init)
#
#         # print all ops, check input/output tensor name.
#         # uncomment it if you donnot know io tensor names.
#         '''
#         print('-------------ops---------------------')
#         op = sess.graph.get_operations()
#         for m in op:
#             try:
#                 # if 'input' in m.values()[0].name:
#                 #     print(m.values())
#                 if m.values()[0].shape.as_list()[1] == 2048: #and (len(m.values()[0].shape.as_list()) == 4):
#                     print(m.values())
#             except:
#                 pass
#         print('-------------ops done.---------------------')
#         '''
#         input_x = sess.graph.get_tensor_by_name('input.1:0')  # input
#         outputs = sess.graph.get_tensor_by_name('502:0')  # 5
#         output_tf_pb = sess.run(outputs, feed_dict={input_x: to_numpy(img)})
#         from ipdb import set_trace;
#
#         set_trace()
#         print('output_tf_pb = {}'.format(output_tf_pb))
