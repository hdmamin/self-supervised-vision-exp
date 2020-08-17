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
        t = self.temperature
        if not t:
            self.temperature = t = int(np.sqrt(x.shape[-1]))
        return x.div(t).softmax(dim=-1)


class Encoder(nn.Module):

    def __init__(self, n=3, c_in=3, fs=(8, 16, 32, 64, 128, 256),
                 strides=(2, 2, 1, 1, 1, 1), kernel_size=3, norm=True,
                 padding=0, act=Mish(), res_blocks=0, **res_kwargs):
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
        super().__init__()
        model = getattr(tvm, arch)(pretrained=True, **kwargs)
        self.model = dict(model.named_children())['features']

    def forward(self, x):
        return self.model(x)


class ClassificationHead(nn.Module, ABC):
    """Abstract class that handles the last activation common to all of our
    classification heads. Subclasses must implement a `_forward` method which
    will be called prior to this activation.
    """

    @valuecheck
    def __init__(self, last_act: ('sigmoid', 'softmax', None) = 'sigmoid',
                 temperature=1.0):
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
        x = self._forward(x_new, x_stack)
        return self.last_act(x)

    @abstractmethod
    def _forward(self, x_new, x_stack):
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
        super().__init__(**act_kwargs)
        self.fc = nn.ModuleList([nn.Linear(f_in, f_out)
                                 for f_in, f_out in zip((f_in, *fs), fs)])
        self.act = act
        self.n_layers = len(fs)

    def _forward(self, x_new, x_stack):
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

