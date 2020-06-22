import unittest
import numpy as np
import os
from glob import glob


class TestFeatureAlign(unittest.TestCase):
    def test_caffe_pytorch_feat_align(self):
        caffe_feat_path = "/export/home/lxy/cvpalgo-fast-reid/tools/deploy/caffe_R50_output"
        pytorch_feat_path = "/export/home/lxy/cvpalgo-fast-reid/demo/logs/R50_256x128_pytorch_feat_output"
        feat_filenames = os.listdir(caffe_feat_path)
        for feat_name in feat_filenames:
            caffe_feat = np.load(os.path.join(caffe_feat_path, feat_name))
            pytorch_feat = np.load(os.path.join(pytorch_feat_path, feat_name))
            sim = np.dot(caffe_feat, pytorch_feat.transpose())[0][0]
            assert sim > 0.97, f"Got similarity {sim} and feature of {feat_name} is not aligned"

    def test_model_performance(self):
        caffe_feat_path = "/export/home/lxy/cvpalgo-fast-reid/tools/deploy/caffe_R50_output"
        feat_filenames = os.listdir(caffe_feat_path)
        feats = []
        for feat_name in feat_filenames:
            caffe_feat = np.load(os.path.join(caffe_feat_path, feat_name))
            feats.append(caffe_feat)
        from ipdb import set_trace; set_trace()



if __name__ == '__main__':
    unittest.main()
