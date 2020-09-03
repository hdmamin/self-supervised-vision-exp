import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch


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

