# encoding: utf-8
"""
@author:  xingyu liao
@contact: sherlockliao01@gmail.com
"""

import torch
import torch.nn.functional as F


# based on:
# https://github.com/kornia/kornia/blob/master/kornia/losses/focal.py

def focal_loss(
        input: torch.Tensor,
        target: torch.Tensor,
        alpha: float,
        gamma: float = 2.0,
        reduction: str = 'mean') -> torch.Tensor:
    r"""Criterion that computes Focal loss.
    See :class:`fastreid.modeling.losses.FocalLoss` for details.
    According to [1], the Focal loss is computed as follows:
    .. math::
        \text{FL}(p_t) = -\alpha_t (1 - p_t)^{\gamma} \, \text{log}(p_t)
    where:
       - :math:`p_t` is the model's estimated probability for each class.
    Arguments:
        alpha (float): Weighting factor :math:`\alpha \in [0, 1]`.
        gamma (float): Focusing parameter :math:`\gamma >= 0`.
        reduction (str, optional): Specifies the reduction to apply to the
         output: ‘none’ | ‘mean’ | ‘sum’. ‘none’: no reduction will be applied,
         ‘mean’: the sum of the output will be divided by the number of elements
         in the output, ‘sum’: the output will be summed. Default: ‘none’.
    Shape:
        - Input: :math:`(N, C, *)` where C = number of classes.
        - Target: :math:`(N, *)` where each value is
          :math:`0 ≤ targets[i] ≤ C−1`.
    Examples:
        >>> N = 5  # num_classes
        >>> loss = FocalLoss(cfg)
        >>> input = torch.randn(1, N, 3, 5, requires_grad=True)
        >>> target = torch.empty(1, 3, 5, dtype=torch.long).random_(N)
        >>> output = loss(input, target)
        >>> output.backward()
    References:
        [1] https://arxiv.org/abs/1708.02002
    """
    if not torch.is_tensor(input):
        raise TypeError("Input type is not a torch.Tensor. Got {}"
                        .format(type(input)))

    if not len(input.shape) >= 2:
        raise ValueError("Invalid input shape, we expect BxCx*. Got: {}"
                         .format(input.shape))

    if input.size(0) != target.size(0):
        raise ValueError('Expected input batch_size ({}) to match target batch_size ({}).'
                         .format(input.size(0), target.size(0)))

    index = torch.where(target != -1)[0]
    target = target[index]
    input = input[index, :]

    n = input.size(0)
    out_size = (n,) + input.size()[2:]
    if target.size()[1:] != input.size()[2:]:
        raise ValueError('Expected target size {}, got {}'.format(
            out_size, target.size()))

    if not input.device == target.device:
        raise ValueError(
            "input and target must be in the same device. Got: {}".format(
                input.device, target.device))

    # compute softmax over the classes axis
    input_soft = F.softmax(input, dim=1)

    # create the labels one hot tensor
    target_one_hot = F.one_hot(target, num_classes=input.shape[1])

    # compute the actual focal loss
    weight = torch.pow(-input_soft + 1., gamma)

    focal = -alpha * weight * torch.log(input_soft)
    loss_tmp = torch.sum(target_one_hot * focal, dim=1)

    if reduction == 'none':
        loss = loss_tmp
    elif reduction == 'mean':
        loss = torch.mean(loss_tmp)
    elif reduction == 'sum':
        loss = torch.sum(loss_tmp)
    else:
        raise NotImplementedError("Invalid reduction mode: {}"
                                  .format(reduction))
    return loss


def binary_focal_loss(inputs, targets, alpha=0.25, gamma=2):
    '''
    Reference: https://github.com/tensorflow/addons/blob/v0.14.0/tensorflow_addons/losses/focal_loss.py
    '''
    if alpha < 0:
        raise ValueError(f'Value of alpha should be greater than or equal to zero, but get {alpha}')
    if gamma < 0:
        raise ValueError(f'Value of gamma should be greater than or equal to zero, but get {gamma}')

    if not torch.is_tensor(inputs):
        raise TypeError("Input type is not a torch.Tensor. Got {}".format(type(inputs)))

    if not len(inputs.shape) >= 2:
        raise ValueError("Invalid input shape, we expect BxCx*. Got: {}".format(inputs.shape))

    if inputs.size(0) != targets.size(0):
        raise ValueError('Expected input batch_size ({}) to match target batch_size ({}).'
                         .format(inputs.size(0), targets.size(0)))

    if not inputs.device == targets.device:
        raise ValueError(
            "input and target must be in the same device. Got: {}".format(
                inputs.device, targets.device))
    
    if len(targets.shape) == 1:
        targets = torch.unsqueeze(targets, 1)

    if targets.dtype != inputs.dtype:
        targets = targets.to(inputs.dtype)

    bce = F.binary_cross_entropy_with_logits(inputs, targets, reduction='none')
    pred_prob = torch.sigmoid(inputs)

    p_t = targets * pred_prob + (1 - targets) * (1 - pred_prob)
    alpha_factor = 1.0
    modulating_factor = 1.0

    if alpha > 0:
        alpha_factor = targets * alpha + (1 - targets) * (1 - alpha)

    if gamma > 0:
        modulating_factor = torch.pow(1.0 - p_t, gamma)

    loss = torch.mean(alpha_factor * modulating_factor * bce)
    return loss
