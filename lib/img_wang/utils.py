import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def summarize_acts(acts):
    print(f'Shape: {acts.shape}')
    flat = acts.flatten().detach().numpy()
    q = np.arange(0, 1.01, .1)
    print('% > 0:', (flat > 0).mean())
    pd.Series(flat).rename('quantiles').quantile(q).pprint()

    plt.hist(flat)
    plt.show()

