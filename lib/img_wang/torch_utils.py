from fastai2.torch_core import show_images
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import random
import torch

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


def top_mistakes(trainer, xb=None, yb=None, ds=None, dl=None, n=16, df=None):
    if xb is None:
        if ds is None:
            ds = trainer.ds_val
            dl = trainer.dl_val
        xb, yb = next(iter(dl))
    if df is None:
        preds = trainer.predict(xb, logits=False)
        # TODO: maybe update collate_fn to assign ds idx to batch idx attr.
        titles = [f'Label: {y.item()}\nPred: {yhat.item():.3f}' #\nIdx: {x.idx}
                  for x, y, yhat in zip(xb, yb, preds)]
        df = pd.DataFrame({'y': yb.squeeze(-1).numpy(),
                           'y_proba': preds.squeeze(-1).cpu().numpy()})
    sorted_mistakes = df.lambda_sort(lambda x: abs(x.y - x.y_proba),
                                     ascending=False)
    idx = sorted_mistakes.index.values
    show_images([ds[i][0] for i in idx[:n]],
                nrows=int(np.ceil(np.sqrt(n))),
                titles=[titles[i] for i in idx[:n]])
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
    assert torch.cuda.is_available(), 'Cuda not available'
    assert DEVICE.type == 'cuda'

