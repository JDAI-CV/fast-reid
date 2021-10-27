# coding: utf-8
"""
Sun, Y. ,  Zheng, L. ,  Yang, Y. ,  Tian, Q. , &  Wang, S. . (2017). Beyond part models: person retrieval with refined part pooling (and a strong convolutional baseline). Springer, Cham.
"""
from typing import Union
import math

import torch
from torch import nn

from fastreid.config import configurable
from fastreid.modeling.meta_arch import Baseline
from fastreid.modeling.meta_arch import META_ARCH_REGISTRY
from fastreid.layers import weights_init_classifier
from fastreid.modeling.losses import cross_entropy_loss, log_accuracy


@META_ARCH_REGISTRY.register()
class PCB(Baseline):
    
    @configurable
    def __init__(
            self,
            *,
            backbone,
            heads,
            pixel_mean,
            pixel_std,
            part_num,
            part_dim,
            embedding_dim,
            loss_kwargs=None
    ):
        """
        NOTE: this interface is experimental.

        Args:
            backbone:
            heads:
            pixel_mean:
            pixel_std:
            part_num
        """
        super(PCB, self).__init__(
                backbone=backbone,
                heads=heads,
                pixel_mean=pixel_mean,
                pixel_std=pixel_std,
                loss_kwargs=loss_kwargs
            )
        self.part_num = part_num
        self.part_dim = part_dim
        self.embedding_dim = embedding_dim
        self.modify_backbone()
        self.random_init()
  
    def modify_backbone(self):
        self.backbone.avgpool_e = nn.AdaptiveAvgPool2d((1, self.part_num))

        # cnn feature
        self.resnet_conv = nn.Sequential(
            self.backbone.conv1, 
            self.backbone.bn1,
            self.backbone.relu,
            self.backbone.maxpool,
            self.backbone.layer1,
            self.backbone.layer2,
            self.backbone.layer3,
            self.backbone.layer4,
        )
        self.layer5 = nn.Sequential(
            self.backbone._make_layer(block=self.backbone.layer4[-1].__class__,
                                      planes=512, blocks=1, stride=2,bn_norm='BN', with_se=False),
            nn.AdaptiveAvgPool2d((1, 1)),
        )

        self.pool_e = nn.Sequential(self.backbone.avgpool_e)

        # embedding
        for i in range(self.part_num):
            name = 'embedder' + str(i)
            setattr(self, name, nn.Linear(self.embedding_dim, self.part_dim))

    @classmethod
    def from_config(cls, cfg):
        config_dict = super(PCB, cls).from_config(cfg)
        config_dict['part_num'] = cfg.MODEL.PCB.PART_NUM
        config_dict['part_dim'] = cfg.MODEL.PCB.PART_DIM
        config_dict['embedding_dim'] = cfg.MODEL.PCB.EMBEDDING_DIM
        return config_dict

    def random_init(self) -> None:
        for m in self.layer5.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                nn.init.normal_(m.weight, 0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        for i in range(self.part_num):
            embedder_i = getattr(self, 'embedder' + str(i))
            embedder_i.apply(weights_init_classifier) 

    def forward(self, batched_inputs):
        # preprocess image
        images = self.preprocess_image(batched_inputs)

        # backbone: extract global features and local features
        features = self.resnet_conv(images)
        features_full = torch.squeeze(self.layer5(features))
        features_part = torch.squeeze(self.pool_e(features))

        embeddings_list = []
        for i in range(self.part_num):
            if self.part_num == 1:
                features_i = features_part
            else:
                features_i = torch.squeeze(features_part[:, :, i])
            
            embedder_i = getattr(self, 'embedder' + str(i))
            embedding_i = embedder_i(features_i)
            embeddings_list.append(embedding_i)
        
        all_features = {'full': features_full, 'parts': embeddings_list}
        if self.training:
            assert "targets" in batched_inputs, "Person ID annotation are missing in training!"
            targets = batched_inputs["targets"]

            # PreciseBN flag, When do preciseBN on different dataset, the number of classes in new dataset
            # may be larger than that in the original dataset, so the circle/arcface will
            # throw an error. We just set all the targets to 0 to avoid this problem.
            if targets.sum() < 0: targets.zero_()

            outputs = self.heads(all_features, targets)
            losses = self.losses(outputs, targets)  # 损失有问题
            return losses
        else:
            outputs = self.heads(all_features)
            return outputs 

    def losses(self, outputs, gt_labels):
        """
        Compute loss from modeling's outputs, the loss function input arguments
        must be the same as the outputs of the model forwarding.
        """
        # model predictions
        pred_class_logits = outputs['pred_class_logits'].detach()
        cls_outputs = outputs['cls_outputs']
        
        # Log prediction accuracy
        log_accuracy(pred_class_logits, gt_labels)

        loss_dict = {}
        loss_names = self.loss_kwargs['loss_names']

        if 'CrossEntropyLoss' in loss_names:
            ce_kwargs = self.loss_kwargs.get('ce')
            loss_dict['loss_cls'] = cross_entropy_loss(
                cls_outputs,
                gt_labels,
                ce_kwargs.get('eps'),
                ce_kwargs.get('alpha')
            ) * ce_kwargs.get('scale')


        return loss_dict
