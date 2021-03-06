from fastai2.torch_core import show_images
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import random
import torch
from torch.utils.data import SequentialSampler, BatchSampler, DataLoader
import warnings

from htools import lmap, add_docstring, select, valuecheck, pd_tools
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


class PredictionExaminer:
    """Examine model predictions. This lets us view rows where the model was
    very confidently wrong, barely wrong, confidently right, barely right, or
    just random rows.
    """

    def __init__(self, trainer):
        self.trainer = trainer
        self.dls = {}
        self.dfs = {}

    def evaluate(self, split='val', return_df=True):
        dl = getattr(self.trainer, f'dl_{split}')
        if 'random' in type(dl.batch_sampler.sampler).__name__.lower():
            dl = DataLoader(dl.dataset, dl.batch_size, shuffle=False,
                            num_workers=dl.num_workers)
        self.dls[split] = dl
        _, y_proba, y_true = self.trainer.validate(dl, True, True,
                                                   logits=False)

        if self.trainer.mode == 'multiclass':
            y_proba, y_pred = y_proba.max(-1)
        else:
            y_pred = (y_proba > self.trainer.thresh).float()

        # Construct title strings.
        titles = [
            f'True: {y.item()}\nPred: {pred.item()} (p={proba.item():.3f})'
            for y, proba, pred in zip(y_true, y_proba, y_pred)
        ]
        df = pd.DataFrame(
            {'y': y_true.squeeze(-1).cpu().numpy(),
             'y_pred': y_pred.squeeze(-1).cpu().numpy(),
             'y_proba': y_proba.squeeze(-1).cpu().numpy(),
             'title': titles}
        )
        df['correct'] = (df.y == df.y_pred)
        # Score of 1 means model was certain of correct answer.
        # Score of -1 means model was certain of incorrect answer.
        # By default, sort with biggest mistakes at top. Don't reset index.
        df['mistake'] = np.where(df.correct, -1, 1) * df.y_proba
        df.sort_values('mistake', ascending=False, inplace=True)
        self.dfs[split] = df
        if return_df: return df.drop('title', axis=1)

    @valuecheck
    def _select_base(self, mode='most_wrong', split='val', n=16,
                     pred_classes=None, true_classes=None, return_df=False):
        """Internal method that provides the functionality for all user-facing
        methods for filtering and displaying results.

        Parameters
        ----------
        mode: str
            One of ('most_wrong', 'least_wrong', 'most_correct',
            'least_correct', 'random'). "wrong/correct" refers to whether the
            predicted class matches the true class, while "most/least"
            considers the model's confidence as well. I.E. "most_wrong" means
            rows where the model predicted the wrong class with high
            confidence.
        n: int
            Number of images to display.
        pred_classes: Iterable[str] or None
            If provided, only show rows where the true label falls into these
            specific classes.
        true_classes: Iterable[str] or None
            If provided, only show rows where the predicted label falls into
            these specific classes.
        return_df: bool

        Returns
        -------
        pd.DataFrame or None: None by default, depends on `return_df`.
        """
        # Filter and sort rows based on desired classes and criteria.
        df, dl = self.dfs[split], self.dls[split]
        if pred_classes is not None:
            if isinstance(pred_classes, int): pred_classes = [pred_classes]
            df = df.loc[df.y_pred.isin(pred_classes)]
        if true_classes is not None:
            if isinstance(true_classes, int): true_classes = [true_classes]
            df = df.loc[df.y.isin(true_classes)]
        if mode == 'most_wrong':
            df = df[~df.correct].sort_values('mistake', ascending=False)
        elif mode == 'least_wrong':
            df = df[~df.correct].sort_values('mistake', ascending=True)
        elif mode == 'most_correct':
            df = df[df.correct].sort_values('mistake', ascending=True)
        elif mode == 'least_correct':
            df = df[df.correct].sort_values('mistake', ascending=False)
        elif mode == 'random':
            df = df.sample(frac=1, replace=False)
        if df.empty:
            warnings.warn('No examples meet that criteria.')
            return
        idx = df.index.values[:n]

        # Display images for selected rows.
        images = [torch.cat(dl.dataset[i][:-1], dim=-1) for i in idx]
        show_images(images, nrows=int(np.ceil(np.sqrt(n))),
                    titles=[df.loc[i, 'title'] for i in idx])
        plt.tight_layout()
        if return_df: return df.dropna('title', axis=1)

    def most_wrong(self, split='val', n=16, pred_classes=None,
                   true_classes=None, return_df=False):
        return self._select_base('most_wrong', split, n, pred_classes,
                                 true_classes, return_df)

    def least_wrong(self, split='val', n=16, pred_classes=None,
                    true_classes=None, return_df=False):
        return self._select_base('least_wrong', split, n, pred_classes,
                                 true_classes, return_df)

    def most_correct(self, split='val', n=16, pred_classes=None,
                     true_classes=None, return_df=False):
        return self._select_base('most_correct', split, n, pred_classes,
                                 true_classes, return_df)

    def least_correct(self, split='val', n=16, pred_classes=None,
                      true_classes=None, return_df=False):
        return self._select_base('least_correct', split, n, pred_classes,
                                 true_classes, return_df)

    def random(self, split='val', n=16, pred_classes=None, true_classes=None,
               return_df=False):
        return self._select_base('random', split, n, pred_classes,
                                 true_classes, return_df)

    def class_to_top_mistakes(self, split='val', n=3):
        df = self.dfs[split]
        return {lbl: dict(df.loc[(~df.correct) & (df.y == lbl),
                                 'y_pred'].value_counts().head(n))
                for lbl in df.y.unique()}

    def confusion_matrix(self, split='val'):
        cm = pd.pivot_table(self.dfs[split], index='y', columns='y_pred',
                            values='title', aggfunc=len, fill_value=0)
        if len(set(cm.shape)) > 1 or not (cm.index == cm.columns).all():
            short_ax = np.argmin(cm.shape)
            cm = cm.reindex(cm.index.values if short_ax == 1
                            else cm.columns.values, axis=short_ax).fillna(0)
        return cm.style.background_gradient(axis=1)

    def label_vcounts(self, split='val', n=None):
        return self.dfs[split].y_pred.vcounts().head(n)

    def pred_vcounts(self, split='val', n=None):
        return self.dfs[split].y.vcounts().head(n)


def top_mistakes(trainer, xb=None, yb=None, dl=None, n=16):
    """Find the biggest mistakes made on a single batch or on a whole dataset
    and displays the corresponding images along with their labels and
    predictions.

    Parameters
    ----------
    trainer: incendio.core.Trainer
    xb: tuple[torch.Tensor]
        Features for a single batch, as obtained by `*xb, yb = next(iter(dl))`.
        If None is provided, we assume you want predictions for a whole
        dataset.
    yb: torch.Tensor
        Labels for a single batch.
    dl: torch.utils.data.DataLoader
        Dataloader to evaluate. If None is provided and no batch is provided
        either, this defaults to the trainer's validation dataloader.
    n: int
        Number of mistakes to display.

    Returns
    -------
    pd.DataFrame: Contains label, predicted class, and predicted probability
    for each example in dataset.
    """
    if xb is None:
        # Handle dataloaders with random shuffling. Need sequential sampling
        # to ensure we associate the right image with its label and prediction.
        if dl and 'random' in type(dl.batch_sampler.sampler).__name__.lower():
            dl = DataLoader(dl.dataset, dl.batch_size, shuffle=False,
                            num_workers=dl.num_workers)
        _, y_proba, y_true = trainer.validate(dl, True, True, logits=False)
    else:
        y_proba = trainer.predict(*xb, logits=False)
        y_true = yb

    if trainer.mode == 'multiclass':
        y_proba, y_pred = y_proba.max(-1)
    else:
        y_pred = y_proba > trainer.thresh

    # Construct title strings.
    titles = []
    for y, proba, pred in zip(y_true, y_proba, y_pred):
        titles.append(
            f'True: {y.item()}\nPred: {pred.item()} (p={proba.item():.3f})'
        )
    df = pd.DataFrame(
        {'y': y_true.squeeze(-1).cpu().numpy(),
         'y_pred': y_pred.cpu().numpy(),
         'y_proba': y_proba.squeeze(-1).cpu().numpy(),
         'title': titles}
    )
    df['correct'] = (df.y == df.y_pred)
    sorted_mistakes = df.lambda_sort(
        lambda x: np.where(x.correct, -1, 1) * x.y_proba, ascending=False
    )
    idx = sorted_mistakes.index.values
    if xb is not None:
        images = [xb[i] for i in idx[:n]]
    else:
        images = [torch.cat(dl.dataset[i][:-1], dim=-1) for i in idx[:n]]
    show_images(images,
                nrows=int(np.ceil(np.sqrt(n))),
                titles=[df.loc[i, 'title'] for i in idx[:n]])
    plt.tight_layout()
    return sorted_mistakes


def reproducible(seed=1, verbose=True):
    """Make training reproducible by setting seeds for numpy, torch, random,
    and python.

    Parameters
    ----------
    seed: int
    verbose: bool
    """
    if verbose: print('Setting seeds for reproducible training.')
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def gpu_setup(make_reproducible=True, seed=1, verbose=True):
    """Check that gpu is available and optionally set random seeds for
    reproducible training.

    Parameters
    ----------
    make_reproducible: bool
        If True, set seeds for random, numpy, torch, and python.
    seed: int
        Used to set all random seeds if make_reproducible is True.
    verbose: bool
        Passed on to `reproducible` function. If a GPU is not available,
        warnings will occur regardless of this parameter since this contains
        important information.
    """
    if make_reproducible: reproducible(seed, verbose)
    if not torch.cuda.is_available(): warnings.warn('Cuda not available.')
    if DEVICE.type != 'cuda':
        print(DEVICE)
        warnings.warn('Incendio device is not cuda.')


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
    """Augmentation function for troubleshooting purposes. Adds noise from a
    random normal distibution with optional truncating (to avoid truncating,
    you can always pass in an enormous max_val and an enormously negative
    min_val).

    Parameters
    ----------
    x: torch.Tensor
        The tensor to adjust.
    mean: int or float
        Mean of random noise distribution.
    std: int or float
        Standard deviation of random noise distribution.
    min_val: int or float
        Outputs will be clamped here. This refers to the output tensor, NOT
        the random noise.
    max_val: int or float
        Outputs will be clamped here. This refers to the output tensor, NOT
        the random noise.

    Returns
    -------
    torch.Tensor: same shape as input x.
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


