import fire
from glob import glob
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import torch

from img_wang.config import Config


def summarize_acts(acts):
    print(f'Shape: {acts.shape}')
    flat = acts.flatten().detach().numpy()
    q = np.arange(0, 1.01, .1)
    print('% > 0:', (flat > 0).mean())
    pd.Series(flat).rename('quantiles').quantile(q).pprint()

    plt.hist(flat)
    plt.show()


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


def next_model_version(new=True):
    """Get model version number (e.g. 4 for version v4). See `next_model_dir`
    for more explanation on how this is used.

    Parameters
    ----------
    new: bool
        If True, get the version number of a completely new model. If False,
        get the highest existing number (use this when `log_cmd` has created
        a subdir in anticipation of an upcoming training run.

    Returns
    -------
    int: Model version number. Differs depending on choice of `new`.
    """
    return max([int(p.stem.strip('v')) for p in Config.model_dir.iterdir()
                if p.is_dir()] + [-1]) + (1 if new else 0)


def next_model_dir(new=True):
    """Get the name of the model subdirectory (e.g. data/models/v4) to save
    training artifacts in. This can be the highest existing model number (for
    the case where our `log_cmd` decorator has created a new directory in
    anticipation of an upcoming training run) or the next subdirectory to
    create (i.e. v{n+1} if the previous case is v{n}). This function does not
    create any directories.

    Parameters
    ----------
    new: bool
        If True, get the name of a completely new model subdir. If False,
        get the name of the existing subdir with the highest model version.

    Returns
    -------
    Path: Subdirectory inside data/models which may or may not exist yet.
    """
    return Config.model_dir/f'v{next_model_version(new)}'


def Display(lines, out):
    """Monkeypatch Fire CLI to print "help" to stdout instead of using `less`
    window. User never calls this with arguments so don't worry about them.
    Eventually add this to spellotape or htools along with the code that does
    the patching:

    import fire
    def main():
        # do something

    if __name__ == '__main__':
        fire.core.Display = Display
        fire.Fire(main)
    """
    out.write('\n'.join(lines) + '\n')


fire.core.Display = Display

