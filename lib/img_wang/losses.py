from functools import partial
import torch
import torch.nn as nn
import torch.nn.functional as F

from htools import identity, valuecheck


class PairwiseLossReduction(nn.Module):
    """Basically lets us use L2 or L1 distance as a loss function with the
    standard reductions. If we don't want to reduce, we could use the built-in
    torch function, but that will usually output a tensor rather than a scalar.
    """

    @valuecheck
    def __init__(self, reduce:('sum', 'mean', 'none')='mean', **kwargs):
        super().__init__()
        self.distance = nn.PairwiseDistance(**kwargs)
        self.reduce = identity if reduce == 'none' else getattr(torch, reduce)

    def forward(self, y_proba, y_true):
        return self.reduce(self.distance(y_proba, y_true))


def contrastive_loss(x1, x2, y, m=1., p=2, reduction='mean'):
    """
    # TODO: find out what a reasonable value for m (margin) is.
    
    Note: 
    
    http://yann.lecun.com/exdb/publis/pdf/hadsell-chopra-lecun-06.pdf
    
    Parameters
    ----------
    x1: torch.Tensor
        Shape (bs, n_features).
    x2: torch.Tensor
        Shape (bs, n_features).
    y: torch.Tensor
        Labels. Unlike the paper, we use the convention that a label of 1 
        means images are similar. This is consistent with all our existing
        datasets and just feels more intuitive.
    m: float
        Margin that prevents dissimilar pairs from affecting the loss unless
        they are sufficiently far apart. I believe the reasonable range of
        values depends on the size of the feature dimension.
    p: int
        The p that determines the p-norm used to calculate the initial 
        distance measure between x1 and x2. The default of 2 therefore uses
        euclidean distance.
    reduction: str
        One of ('sum', 'mean', 'none'). Standard pytorch loss reduction. Keep
        in mind 'none' will probably not allow backpropagation since it
        returns a rank 2 tensor.
        
    Returns
    -------
    torch.Tensor: Scalar measuring the contrastive loss. If no reduction is
    applied, this will instead be a tensor of shape (bs,).
    """
    reduction = identity if reduction == 'none' else getattr(torch, reduction)
    dw = F.pairwise_distance(x1, x2, p, keepdim=True) 
    # Loss_similar + Loss_different
    res = y*dw.pow(p).div(2) + (1-y)*torch.clamp_min(m-dw, 0).pow(p).div(2)
    return reduction(res)


class ContrastiveLoss1d(nn.Module):
    
    def __init__(self, m=1., p=2, reduction='mean'):
        super().__init__()
        self.m = m
        self.p = p
        self.reduction = reduction
        self.loss = partial(contrastive_loss, m=m, p=p, reduction=reduction)
        
    def forward(self, x1, x2, y_true):
        return self.loss(x1, x2, y_true)


class ContrastiveLoss2d(nn.Module):
    
    def __init__(self, m=1., p=2, reduction='mean'):
        super().__init__()
        self.m = m
        self.p = p
        self.loss = partial(contrastive_loss, m=m, p=p, reduction='none')
        
        if reduction == 'none':
            self.reduction = identity
        elif reduction == 'row':
            self.reduction = partial(torch.sum, dim=-1)
        else:
            self.reduction = getattr(torch, reduction)
        
    def forward(self, x1, x2, y_true):
        # x1 has shape (bs, feats). x2 has shape (bs, n_item, n_feats).
        # I.E. we're comparing 1 image to `n_item` variants.
        # y_true has shape (bs, n_item).
        # Basically multi-label classification with OHE labels.
        # Output is scalar if reduction is 'mean' or 'sum', same shape as y
        # if reduction is 'none', or shape (bs,) if reduction is 'row'.
        bs, n, dim = x2.shape
        res = self.loss(x1.repeat_interleave(n, dim=0), 
                        x2.view(-1, dim),
                        y_true.view(-1, 1))
        return self.reduction(res.view(bs, -1))


