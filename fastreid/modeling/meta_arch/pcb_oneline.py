# coding: utf-8
"""
Sun, Y. ,  Zheng, L. ,  Yang, Y. ,  Tian, Q. , &  Wang, S. . (2017). Beyond part models: person retrieval with refined part pooling (and a strong convolutional baseline). Springer, Cham.
实现和线上一模一样的PCB
"""
import torch

from fastreid.modeling.losses import cross_entropy_loss, log_accuracy
from fastreid.modeling.meta_arch import Baseline
from fastreid.modeling.meta_arch import META_ARCH_REGISTRY


@META_ARCH_REGISTRY.register()
class PcbOnline(Baseline):

    def forward(self, batched_inputs):
        # TODO: 减均值除方差

        q = batched_inputs['q_img']
        x = batched_inputs['p_img']
        qf = self.cnn(q)
        xf = self.cnn(x)
        # L2 norm  TODO: 查看具体数值，确定他到底是什么范数
        qf = self._norm(qf)
        xf = self._norm(xf)



        qf_full, qf_part_0, qf_part_1, qf_part_2 = torch.split(qf, [2048, 512, 512, 512], dim=-1)
        xf_full, xf_part_0, xf_part_1, xf_part_2 = torch.split(xf, [2048, 512, 512, 512], dim=-1)

        m_full = self.dense_matching_full(
            torch.cat([qf_full, xf_full, (qf_full - xf_full).abs(), qf_full * xf_full], dim=-1))
        m_part_0 = self.dense_matching_part_0(
            torch.cat([qf_part_0, xf_part_0, (qf_part_0 - xf_part_0).abs(), qf_part_0 * xf_part_0], dim=-1))
        m_part_1 = self.dense_matching_part_1(
            torch.cat([qf_part_1, xf_part_1, (qf_part_1 - xf_part_1).abs(), qf_part_1 * xf_part_1], dim=-1))
        m_part_2 = self.dense_matching_part_2(
            torch.cat([qf_part_2, xf_part_2, (qf_part_2 - xf_part_2).abs(), qf_part_2 * xf_part_2], dim=-1))

        m_all = self.dense_matching_all(torch.cat([m_full, m_part_0, m_part_1, m_part_2], dim=-1))

        return m_all.squeeze()

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
