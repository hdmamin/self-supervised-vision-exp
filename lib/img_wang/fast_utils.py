from fastai2.vision.all import fastuple, Transform, show_image, get_grid, \
    typedispatch
from functools import partial
import numpy as np
import torch

from img_wang.data import ImageMixer, load_img


class MixedImages(fastuple):
    """1 sample (inputs and outputs) from FastMixupDataset."""

    def show(self, ctx=None, **kwargs):
        *imgs, label = self
        return show_image(torch.cat(imgs, dim=2),
                          title=label.numpy().round(3).tolist(),
                          ctx=ctx, **kwargs)


class FastMixupDataset(Transform):
    """Basically fastai equivalent of img_wang.data.MixupDataset."""

    def __init__(self, paths, shape=(160, 160), n=3, **kwargs):
        """Saw a bug once with n=2 but couldn't reproduce it since then. Not
        sure if it was because of n or if it was unrelated.

        Parameters
        ----------
        paths
        shape
        n
        kwargs
        """
        self.paths = np.array(paths)
        self.n = n
        self.mixer = ImageMixer(n=n, **kwargs)
        self.load_img = partial(load_img, shape=shape)

    def encodes(self, i):
        remaining = len(self.paths) - i
        # Fastai transform doesn't define __len__ so we need another way of
        # handling when index is close to the max_idx.
        if remaining >= self.n:
            paths = self.paths[i:i + self.n]
        else:
            paths = self.paths[np.r_[i:i + remaining, 0:self.n - remaining]]
        images = map(self.load_img, paths)
        xb, yb = self.mixer.transform(*images)
        return MixedImages(*xb, yb)


@typedispatch
def show_batch(x:MixedImages, y, samples, ctxs=None, max_n=6, nrows=None,
               ncols=2, figsize=None, **kwargs):
    """Patch to allow dataloaders `show_batch` method to work. User never calls
    this directly so I feel okay leaving parameters undocumented. One note as a
    reminder to myself: fastai2 passes a whole batch in as x for some reason.
    So x is a tuple where the first n-1 items are (bs, c, h, w) tensors and the
    nth item is a (bs, n-1) tensor of labels.
    """
    if figsize is None:
        figsize = (ncols*6, max_n//ncols * 3)
    if ctxs is None:
        nrows = min(x[0].shape[0], max_n)
        ctxs = get_grid(nrows, nrows=nrows, ncols=ncols,
                        figsize=figsize)
    for i, ctx in enumerate(ctxs):
        MixedImages([row[i] for row in x]).show(ctx=ctx, figsize=(12, 5))

