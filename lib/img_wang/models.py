from abc import ABC, abstractmethod
from fastai2.layers import PoolFlatten
import numpy as np
import torch
import torch.nn as nn
from torchvision import models as tvm
import warnings

from htools import valuecheck, identity
from incendio.core import BaseModel
from incendio.layers import Mish, ConvBlock, ResBlock


class SmoothSoftmax(nn.Module):
    """Softmax with temperature baked in."""

    def __init__(self, temperature='auto'):
        """
        Parameters
        ----------
        temperature: float or str
            If a float, this is the temperature to divide activations by before
            applying the softmax. Values larger than 1 soften the distribution
            while values between 0 and 1 sharpen it. If str ('auto'), this will
            compute the square root of the last dimension of x's shape the
            first time the forward method is called and use that for subsequent
            calls.
        """
        super().__init__()
        self.temperature = None if temperature == 'auto' else temperature

    def forward(self, x):
        # Kind of silly but this is called every mini batch so removing an
        # extra dot attribute access saves a little time.
        while True:
            try:
                return x.div(self.temperature).softmax(dim=-1)
            except TypeError:
                self.temperature = int(np.sqrt(x.shape[-1]))


class Encoder(nn.Module):

    def __init__(self, n=3, c_in=3, fs=(8, 16, 32, 64, 128, 256),
                 strides=(2, 2, 1, 1, 1, 1), kernel_size=3, norm=True,
                 padding=0, act=Mish(), res_blocks=0, **res_kwargs):
        """TODO: encoder docs

        Parameters
        ----------
        n
        c_in
        fs
        strides
        kernel_size
        norm
        padding
        act
        res_blocks
        res_kwargs
        """
        super().__init__()
        if len(fs) != len(strides):
            raise ValueError('Strides and f_dims must be same length (1 '
                             ' value for each conv block).')

        self.n = n
        self.conv = nn.Sequential(
            *[ConvBlock(f_in, f_out, kernel_size=kernel_size, norm=norm,
                        activation=act, stride=stride, padding=padding)
              for f_in, f_out, stride in zip((c_in, *fs), fs, strides)]
        )
        if res_blocks:
            self.res = nn.Sequential(
                *[ResBlock(c_in=fs[-1], activation=act, **res_kwargs)
                  for _ in range(res_blocks)]
            )

    def forward(self, x):
        x = self.conv(x)                           # (bs, f[-1], new_h, new_w)
        if hasattr(self, 'res'): x = self.res(x)   # No change in dims.
        return x


class TorchvisionEncoder(nn.Module):
    """Create an encoder from a standard architecture provided by Torchvision.
    By default, pretrained weights will be used but that can be overridden in
    the constructor.
    """

    def __init__(self, arch='mobilenet_v2', **kwargs):
        """Create an encoder from a pretrained torchvision model.

        Parameters
        ----------
        arch: str
            Name of architecture to use. See options:
            https://pytorch.org/docs/stable/torchvision/models.html
        kwargs: any
            Addition kwargs will be forwarded to the model constructor.
        """
        super().__init__()
        model = getattr(tvm, arch)(pretrained=True, **kwargs)
        self.model = dict(model.named_children())['features']

    def forward(self, x):
        """We return features only so this should have a shape like
        (bs, feature_dim, new height, new width).
        """
        return self.model(x)


class ClassificationHead(nn.Module, ABC):
    """Abstract class that handles the last activation common to all of our
    classification heads. Subclasses must implement a `_forward` method which
    will be called prior to this activation. This is arguably overkill in terms
    of how much abstraction we really need but it was getting annoying copying
    the `last_act` code over to every head and I want to make it really easy
    to experiment.
    """

    @valuecheck
    def __init__(self, last_act: ('sigmoid', 'softmax', None) = 'sigmoid',
                 temperature=1.0):
        """
        Parameters
        ----------
        last_act: str or None
            Determines what the final activation will be. In regression mode,
            both sigmoid and softmax are viable options. In classification
            mode, use None.
        temperature: float or str
            Passed to SmoothSoftmax if `last_act` is 'softmax'. Larger values
            incentivize less extreme predictions (e.g. near .5 rather than .99)
            while values < 1 incentivize predictions near 0 or 1. The only
            acceptable str is 'auto' which will compute the square root of x's
            feature dimension.
        """
        super().__init__()
        if last_act == 'softmax':
            self.last_act = SmoothSoftmax(temperature)
        else:
            warnings.warn('Temperature is ignored when last activation is '
                          'not softmax.')
            if last_act == 'sigmoid':
                self.last_act = torch.sigmoid
            if last_act is None:
                self.last_act = identity

    def forward(self, x_new, x_stack):
        """
        Parameters
        ----------
        x_new: torch.tensor
            The tensor encoding a single image. This is after pooling so we
            have shape (bs, feature_dim).
        x_stack: torch.tensor
            Tensor encoding the n source images. We have shape
            (bs, n, feature_dim).

        Returns
        -------
        torch.tensor: This will have shape (bs, n) regardless of whether we're
        in classification mode or regression mode.
        """
        x = self._forward(x_new, x_stack)
        return self.last_act(x)

    @abstractmethod
    def _forward(self, x_new, x_stack):
        """Child classes must implement this method. This does everything
        except for the final activation.
        """
        raise NotImplementedError


class DotProductHead(ClassificationHead):
    """Simple classification head that essentially computes dot products
    between the constructed image vector and each of the source image vectors
    (think of this as an unscaled cosine similarity). This may be too simple
    an approach: it asks a lot of the encoder network.
    """

    def __init__(self, **act_kwargs):
        super().__init__(**act_kwargs)

    def _forward(self, x_new, x_stack):
        return (x_new[:, None, ...] * x_stack).sum(-1)


class MLPHead(ClassificationHead):
    """Classification head that uses 1 or more linear layers after the encoder.
    """

    @valuecheck
    def __init__(self, f_in, fs=(256, 1), act=Mish(), **act_kwargs):
        """
        Parameters
        ----------
        f_in
        fs
        act: callable
            Can be a nn.Module or something from torch.nn.functional. It's
            called after each linear layer except the last one (that is handled
            by `last_act` in the parent class).
        act_kwargs: any
            Passed on to parent class to create the final activation function.
            Available options are `last_act` and `temperature`.
        """
        super().__init__(**act_kwargs)
        self.fc = nn.ModuleList([nn.Linear(f_in, f_out)
                                 for f_in, f_out in zip((f_in, *fs), fs)])
        self.act = act
        self.n_layers = len(fs)

    def _forward(self, x_new, x_stack):
        """We start with the same element-wise multiple as in `DotProductHead`.
        However, instead of simply summing and returning, we follow this with
        a stack of linear layers. This gives the user the flexibility to
        increase/decrease the dimension as much as they want before ultimately
        outputting a tensor of shape (bs, n). Notice the final linear layer
        has 1 output node: this is squeezed out so we have one tensor of
        predictions per row.
        """
        x = x_new[:, None, ...] * x_stack
        for i, layer in enumerate(self.fc, 1):
            x = layer(x)
            if i < self.n_layers:
                x = self.act(x)
        return x.squeeze(-1)


class Unmixer(BaseModel):
    """Model wrapper that contains an encoder, pooling layer, and
    classification head. This is intended for use on the unsupervised task.
    """

    def __init__(self, encoder=None, head=None):
        """
        Parameters
        ----------
        encoder: nn.Module
            Outputs a tensor of shape (bs, channels, h, w). Unmixer will
            apply adaptive concat pooling and flatten the output to shape
            (bs, channels*2).
        head: nn.Module
            Accepts two inputs (x_new and x_stack) and outputs a tensor of
            predicted probabilities of shape (bs, n_src_imgs).
        """
        super().__init__()
        self.encoder = encoder or Encoder()
        self.pool = PoolFlatten('cat')
        self.head = head or DotProductHead()

    def forward(self, *xb):
        """For now, we naively apply the encoder n+1 times in sequence. See
        nb02 UnmixerModel for an example of how to reshape the input to process
        all n+1 images at once. Still need to confirm whether that
        implementation works correctly, though.
        """
        x_new, *x = [self.pool(self.encoder(x)) for x in xb]
        x = torch.stack(x, dim=1)
        return self.head(x_new, x)


class PairwiseLossReduction(nn.Module):
    """Basically lets us use L2 or L1 distance as a loss function with the
    standard reductions. If we don't want to reduce, we could use the built-in
    torch function, but that will usually output a tensor rather than a scalar.
    """

    @valuecheck
    def __init__(self, reduce: ('sum', 'mean') = 'mean', **kwargs):
        super().__init__()
        self.distance = nn.PairwiseDistance(**kwargs)
        self.reduce = getattr(torch, reduce)

    def forward(self, y_proba, y_true):
        return self.reduce(self.distance(y_proba, y_true))


class SupervisedEncoderClassifier(nn.Module):
    """Simple model for supervised task. This was mostly developed for quick
    troubleshooting while investigating issues with the unsupervised task,
    but ideally it will be flexible enough that we can still use it when it
    comes time to transfer from the pretraining task.
    """

    def __init__(self, enc=None, n_classes=20):
        """
        Parameters
        ----------
        enc: nn.Module
            A model that accepts a batch of single images (as opposed to the
            supervised task which takes in n+1 images per row) and outputs a
            tensor of shape (bs, feature_dim, new_height, new_width). Pooling
            will occur afterwards so this shouldn't be a vector.
        n_classes: int
            Number of output classes in the supervised task. Imagewang uses
            a training set with 20 classes (multiclass, single label), so even
            though the validation set has only 10 classes, we need to be
            capable of predicting all 20.
        """
        super().__init__()
        self.n_classes = n_classes

        # Layers
        self.enc = enc or Encoder()
        self.pool = PoolFlatten('cat')
        # Concat pool doubles last feature dimension.
        self.fc = nn.Linear(list(self.enc.parameters())[-1].shape[0] * 2,
                            n_classes)

    def forward(self, x):
        """
        Parameters
        ----------
        x: torch.tensor
            Batch of images with shape (bs, 3, height, width).

        Returns
        -------
        torch.tensor: Logits with shape (bs, n_classes). These have NOT been
        passed through a final activation function.
        """
        x = self.enc(x)
        x = self.pool(x)
        return self.fc(x).squeeze()
