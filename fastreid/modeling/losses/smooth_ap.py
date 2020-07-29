# encoding: utf-8
"""
@author:  xingyu liao
@contact: sherlockliao01@gmail.com
"""

# based on:
# https://github.com/Andrew-Brown1/Smooth_AP/blob/master/src/Smooth_AP_loss.py

import torch
import torch.nn.functional as F

from fastreid.utils import comm
from fastreid.modeling.losses.utils import concat_all_gather


def sigmoid(tensor, temp=1.0):
    """ temperature controlled sigmoid
    takes as input a torch tensor (tensor) and passes it through a sigmoid, controlled by temperature: temp
    """
    exponent = -tensor / temp
    # clamp the input tensor for stability
    exponent = torch.clamp(exponent, min=-50, max=50)
    y = 1.0 / (1.0 + torch.exp(exponent))
    return y


class SmoothAP(object):
    r"""PyTorch implementation of the Smooth-AP loss.
    implementation of the Smooth-AP loss. Takes as input the mini-batch of CNN-produced feature embeddings and returns
    the value of the Smooth-AP loss. The mini-batch must be formed of a defined number of classes. Each class must
    have the same number of instances represented in the mini-batch and must be ordered sequentially by class.
    e.g. the labels for a mini-batch with batch size 9, and 3 represented classes (A,B,C) must look like:
        labels = ( A, A, A, B, B, B, C, C, C)
    (the order of the classes however does not matter)
    For each instance in the mini-batch, the loss computes the Smooth-AP when it is used as the query and the rest of the
    mini-batch is used as the retrieval set. The positive set is formed of the other instances in the batch from the
    same class. The loss returns the average Smooth-AP across all instances in the mini-batch.
    Args:
        anneal : float
            the temperature of the sigmoid that is used to smooth the ranking function. A low value of the temperature
            results in a steep sigmoid, that tightly approximates the heaviside step function in the ranking function.
        batch_size : int
            the batch size being used during training.
        num_id : int
            the number of different classes that are represented in the batch.
        feat_dims : int
            the dimension of the input feature embeddings
    Shape:
        - Input (preds): (batch_size, feat_dims) (must be a cuda torch float tensor)
        - Output: scalar
    Examples::
        >>> loss = SmoothAP(0.01, 60, 6, 256)
        >>> input = torch.randn(60, 256, requires_grad=True).cuda()
        >>> output = loss(input)
        >>> output.backward()
    """

    def __init__(self, cfg):
        r"""
        Parameters
        ----------
        cfg: (cfgNode)

        anneal : float
            the temperature of the sigmoid that is used to smooth the ranking function
        batch_size : int
            the batch size being used
        num_id : int
            the number of different classes that are represented in the batch
        feat_dims : int
            the dimension of the input feature embeddings
        """

        self.anneal = 0.01
        self.num_id = cfg.SOLVER.IMS_PER_BATCH // cfg.DATALOADER.NUM_INSTANCE
        # self.num_id = 6

    def __call__(self, embedding, targets):
        """Forward pass for all input predictions: preds - (batch_size x feat_dims) """

        # ------ differentiable ranking of all retrieval set ------
        embedding = F.normalize(embedding, dim=1)

        feat_dim = embedding.size(1)

        # For distributed training, gather all features from different process.
        if comm.get_world_size() > 1:
            all_embedding = concat_all_gather(embedding)
            all_targets = concat_all_gather(targets)
        else:
            all_embedding = embedding
            all_targets = targets

        sim_dist = torch.matmul(embedding, all_embedding.t())
        N, M = sim_dist.size()

        # Compute the mask which ignores the relevance score of the query to itself
        mask_indx = 1.0 - torch.eye(M, device=sim_dist.device)
        mask_indx = mask_indx.unsqueeze(dim=0).repeat(N, 1, 1)  # (N, M, M)

        # sim_dist -> N, 1, M -> N, M, N
        sim_dist_repeat = sim_dist.unsqueeze(dim=1).repeat(1, M, 1)  # (N, M, M)
        # sim_dist_repeat_t = sim_dist.t().unsqueeze(dim=1).repeat(1, N, 1)  # (N, N, M)

        # Compute the difference matrix
        sim_diff = sim_dist_repeat - sim_dist_repeat.permute(0, 2, 1)  # (N, M, M)

        # Pass through the sigmoid
        sim_sg = sigmoid(sim_diff, temp=self.anneal) * mask_indx

        # Compute all the rankings
        sim_all_rk = torch.sum(sim_sg, dim=-1) + 1  # (N, N)

        pos_mask = targets.view(N, 1).expand(N, M).eq(all_targets.view(M, 1).expand(M, N).t()).float()  # (N, M)

        pos_mask_repeat = pos_mask.unsqueeze(1).repeat(1, M, 1)  # (N, M, M)

        # Compute positive rankings
        pos_sim_sg = sim_sg * pos_mask_repeat
        sim_pos_rk = torch.sum(pos_sim_sg, dim=-1) + 1  # (N, N)

        # sum the values of the Smooth-AP for all instances in the mini-batch
        ap = 0
        group = N // self.num_id
        for ind in range(self.num_id):
            pos_divide = torch.sum(
                sim_pos_rk[(ind * group):((ind + 1) * group), (ind * group):((ind + 1) * group)] / (sim_all_rk[(ind * group):((ind + 1) * group), (ind * group):((ind + 1) * group)]))
            ap += pos_divide / torch.sum(pos_mask[ind*group]) / N
        return 1 - ap


class SmoothAP_old(torch.nn.Module):
    """PyTorch implementation of the Smooth-AP loss.
    implementation of the Smooth-AP loss. Takes as input the mini-batch of CNN-produced feature embeddings and returns
    the value of the Smooth-AP loss. The mini-batch must be formed of a defined number of classes. Each class must
    have the same number of instances represented in the mini-batch and must be ordered sequentially by class.
    e.g. the labels for a mini-batch with batch size 9, and 3 represented classes (A,B,C) must look like:
        labels = ( A, A, A, B, B, B, C, C, C)
    (the order of the classes however does not matter)
    For each instance in the mini-batch, the loss computes the Smooth-AP when it is used as the query and the rest of the
    mini-batch is used as the retrieval set. The positive set is formed of the other instances in the batch from the
    same class. The loss returns the average Smooth-AP across all instances in the mini-batch.
    Args:
        anneal : float
            the temperature of the sigmoid that is used to smooth the ranking function. A low value of the temperature
            results in a steep sigmoid, that tightly approximates the heaviside step function in the ranking function.
        batch_size : int
            the batch size being used during training.
        num_id : int
            the number of different classes that are represented in the batch.
        feat_dims : int
            the dimension of the input feature embeddings
    Shape:
        - Input (preds): (batch_size, feat_dims) (must be a cuda torch float tensor)
        - Output: scalar
    Examples::
        >>> loss = SmoothAP(0.01, 60, 6, 256)
        >>> input = torch.randn(60, 256, requires_grad=True).cuda()
        >>> output = loss(input)
        >>> output.backward()
    """

    def __init__(self, anneal, batch_size, num_id, feat_dims):
        """
        Parameters
        ----------
        anneal : float
            the temperature of the sigmoid that is used to smooth the ranking function
        batch_size : int
            the batch size being used
        num_id : int
            the number of different classes that are represented in the batch
        feat_dims : int
            the dimension of the input feature embeddings
        """
        super().__init__()

        assert(batch_size%num_id==0)

        self.anneal = anneal
        self.batch_size = batch_size
        self.num_id = num_id
        self.feat_dims = feat_dims

    def forward(self, preds):
        """Forward pass for all input predictions: preds - (batch_size x feat_dims) """

        preds = F.normalize(preds, dim=1)
        # ------ differentiable ranking of all retrieval set ------
        # compute the mask which ignores the relevance score of the query to itself
        mask = 1.0 - torch.eye(self.batch_size)
        mask = mask.unsqueeze(dim=0).repeat(self.batch_size, 1, 1)
        # compute the relevance scores via cosine similarity of the CNN-produced embedding vectors
        sim_all = torch.mm(preds, preds.t())
        sim_all_repeat = sim_all.unsqueeze(dim=1).repeat(1, self.batch_size, 1)
        # compute the difference matrix
        sim_diff = sim_all_repeat - sim_all_repeat.permute(0, 2, 1)
        # pass through the sigmoid
        sim_sg = sigmoid(sim_diff, temp=self.anneal) * mask
        # compute the rankings
        sim_all_rk = torch.sum(sim_sg, dim=-1) + 1

        # ------ differentiable ranking of only positive set in retrieval set ------
        # compute the mask which only gives non-zero weights to the positive set
        xs = preds.view(self.num_id, int(self.batch_size / self.num_id), self.feat_dims)
        pos_mask = 1.0 - torch.eye(int(self.batch_size / self.num_id))
        pos_mask = pos_mask.unsqueeze(dim=0).unsqueeze(dim=0).repeat(self.num_id, int(self.batch_size / self.num_id), 1, 1)
        # compute the relevance scores
        sim_pos = torch.bmm(xs, xs.permute(0, 2, 1))
        sim_pos_repeat = sim_pos.unsqueeze(dim=2).repeat(1, 1, int(self.batch_size / self.num_id), 1)
        # compute the difference matrix
        sim_pos_diff = sim_pos_repeat - sim_pos_repeat.permute(0, 1, 3, 2)
        # pass through the sigmoid
        sim_pos_sg = sigmoid(sim_pos_diff, temp=self.anneal) * pos_mask
        # compute the rankings of the positive set
        sim_pos_rk = torch.sum(sim_pos_sg, dim=-1) + 1

        # sum the values of the Smooth-AP for all instances in the mini-batch
        ap = torch.zeros(1)
        group = int(self.batch_size / self.num_id)
        for ind in range(self.num_id):
            pos_divide = torch.sum(sim_pos_rk[ind] / (sim_all_rk[(ind * group):((ind + 1) * group), (ind * group):((ind + 1) * group)]))
            ap = ap + ((pos_divide / group) / self.batch_size)

        return 1-ap

if __name__ == '__main__':
    loss1 = SmoothAP(0.01)
    loss2 = SmoothAP_old(0.01, 60, 6, 256)

    inputs = torch.randn(60, 256, requires_grad=True)
    targets = []
    for i in range(6):
        targets.extend([i]*10)
    targets = torch.LongTensor(targets)

    output1 = loss1(inputs, targets)
    output2 = loss2(inputs)

    print(torch.sum(output1 - output2))
