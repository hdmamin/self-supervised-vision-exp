from fastai2.torch_core import show_images
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import random
import torch
import warnings

from htools import lmap
from incendio.core import DEVICE


def rand_choice(tn):
    """Like np.random.choice but for tensors. For now this only works for
    selecting single values from rank 1 tensors.

    Parameters
    ----------
    tn: torch.tensor
        Shape (n,). Any dtype.

    Returns
    -------
    torch.tensor: Rank 1, shape (1,).
    """
    assert tn.ndim == 1, 'rand_choice is designed to work on rank 1 tensors.'
    idx = torch.randint(tn.shape[0], size=(1,))
    return tn[idx]


def summarize_acts(acts):
    print(f'Shape: {acts.shape}')
    flat = acts.flatten().detach().numpy()
    q = np.arange(0, 1.01, .1)
    print('% > 0:', (flat > 0).mean())
    pd.Series(flat).rename('quantiles').quantile(q).pprint()

    plt.hist(flat)
    plt.show()


def top_mistakes(trainer, xb=None, yb=None, dl=None, n=16, df=None):
    if xb is None:
        if dl is None: dl = trainer.dl_val
        xb, yb = next(iter(dl))
    if df is None:
        # Construct title strings.
        y_proba = trainer.predict(xb, logits=False)
        if trainer.mode == 'multiclass':
            y_proba, y_pred = y_proba.max(-1)
        else:
            y_pred = y_proba > trainer.threshold
        titles = []
        idx = getattr(xb, 'idx', [None] * len(xb))
        for x, y, proba, pred, i in zip(xb, yb, y_proba, y_pred, idx):
            titles.append(
                f'Label: {y.item()}\nPred: {pred.item()} '
                f'(p={proba.item():.3f})\nIdx: {i}'
            )

        df = pd.DataFrame(
            {'y': yb.squeeze(-1).numpy(),
             'y_pred': y_pred.cpu().numpy(),
             'y_proba': y_proba.squeeze(-1).cpu().numpy(),
             'title': titles}
        )
    sorted_mistakes = df.lambda_sort(lambda x: (x.y != x.y_pred) * x.y_proba,
                                     ascending=False)
    idx = sorted_mistakes.index.values
    show_images([xb[i] for i in idx[:n]],
                nrows=int(np.ceil(np.sqrt(n))),
                titles=[df.loc[i, 'title'] for i in idx[:n]])
    plt.tight_layout()
    return sorted_mistakes


def reproducible(seed=1, verbose=True):
    if verbose: print('Setting seeds for reproducible training.')
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def gpu_setup(make_reproducible=True, seed=1, verbose=1):
    if make_reproducible: reproducible(seed, verbose)
    if not torch.cuda.is_available(): warnings.warn('Cuda not available.')
    if DEVICE.type != 'cuda': warnings.warn('Incendio device is not cuda.')


def flip_tensor(x, dim=-1):
    """Flip a tensor along a specified dimension.

    Parameters
    ----------
    x: torch.tensor
    dim: int
        For a tensor image, the default flips it horizontally.

    Returns
    -------
    torch.tensor
    """
    idx = [slice(None) for _ in x.shape]
    idx[dim] = range(x.shape[dim]-1, -1, -1)
    return x[idx]


def random_noise(x, mean=0, std=1, min_val=0, max_val=1):
    """TODO: docs

    Parameters
    ----------
    x
    mean
    std
    min_val
    max_val

    Returns
    -------

    """
    return torch.clamp(x + torch.randn_like(x).mul(std).add(mean),
                       min_val, max_val)


def n_out_channels(model):
    """Try to guess number of output channels of a CNN encoder model. Not sure
    how well this logic holds up so I suspect it could lead to bugs if not used
    carefully. I already know it doesn't work on fastai's classification head,
    so we can't use it on the whole encoder-decoder model - just the encoder.

    Parameters
    ----------
    model: nn.Module
        Model where last layer is Conv2d (or BatchNorm2d) following a conv
        layer.

    Returns
    -------
    int: Number of output channels. In other words, if we pool and flatten,
    we'll get a shape like (bs, {return_value}). If we use concat pooling,
    {return_value} will need to be multiplied by 2.
    """
    return list(model.parameters())[-1].shape[-1]


