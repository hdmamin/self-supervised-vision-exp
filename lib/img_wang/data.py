from copy import copy
from fastai2.torch_core import TensorImage
from fastai2.data.transforms import get_image_files
from fastai2.vision.core import load_image
from functools import partial
import numpy as np
from sklearn.model_selection import train_test_split
import torch
from torch.utils.data import Dataset, DataLoader

from htools import Args, valuecheck


class ImageMixer:
    """The transformation that powers MixupDS.

    Inspired by the "Visual-Spatial - Entangled Figures" task here:
    http://www.happy-neuron.com/brain-games/visual-spatial/entangled-figures

    The key idea: when a human plays the happy neuron task, it is much easier
    when viewing an entanglement of objects they recognize (e.g. bicycle, leaf,
    etc.) than of random scribbles. I noticed that for the harder levels, my
    strategy was to try to quickly identify a couple distinctive features, then
    search for them in the individual images once they appeared. This quick
    feature extraction seems very close to what we want to achieve during the
    pre-training step.
    """

    def __init__(self, n=3, a=5, b=8, dist=None):
        """
        Parameters
        ----------
        n: int
            Number of images to use as inputs. With the current implementation,
            the constructed image will use exactly 2 of these. The rest will
            be negatives (zero weight).
        a: int
            Parameter in beta distribution.
        b: int
            Parameter in beta distribution.
        dist: torch.distribution
            This can be anything with a "sample()" method that generates a
            random value between 0 and 1. By default, we use a Beta
            distribution. If one is passed in, a and b are ignored.
        """
        assert n >= 2, 'n must be >=2 so we can combine images.'

        self.dist = dist or torch.distributions.beta.Beta(a, b)
        self.n = n

    def transform(self, *images):
        """Create linear combination of images.

        Parameters
        ----------
        images: torch.tensors

        Returns
        -------
        tuple(torch.tensor): First item is (n_channel, h, w*self.n), meaning
        images are horizontally stacked. The first of these is the new image.
        The second item is the rank 1 tensor of weights used to generate the
        combination. These will serve as labels in our self-supervised task.
        """
        w = self._generate_weights()
        return (self._combine_images(images, w), *images), w

    def _generate_weights(self):
        """
        Returns
        -------
        weights: torch.Tensor
            Vector of length self.n. Exactly 2 of these values are
            nonzero and they sum to 1. This will be used to compute a linear
            combination of a row of images.
        """
        weights = np.zeros(self.n)
        p = self.dist.sample()
        indices = np.random.choice(self.n, size=2, replace=False)
        weights[indices] = p, 1 - p
        return torch.tensor(weights, dtype=torch.float)

    def _combine_images(self, images, weights):
        """Create linear combination of multiple images.

        Parameters
        ----------
        images: torch.Tensor
        weights: torch.Tensor
            Vector with 1 value for each image. Exactly 2 of these values are
            nonzero and they sum to 1. I.e. if we have 3 images a, b, and c,
            weights would look something like [0, .3, .7].

        Returns
        -------
        torch.tensor: Shape (channels, height, width), same as each of the
            input images.
        """
        images = torch.stack(images, dim=0)
        # 3 new dimensions correspond to (c, h, w), NOT self.n.
        return (weights[:, None, None, None] * images).sum(0).float()


class MixupDataset(Dataset):
    """Dataset for unsupervised learning. We create a new image that is a
    linear combination of two other images. The network is shown the
    constructed image, the two sources images, and 1 or more other images that
    weren't used to construct the new image. It's tasked with predicting
    the weights that correspond to each of these source images when computing
    the linear combination.
    """

    def __init__(self, dir_=None, paths=(), shape=(128, 128), n=3,
                 regression=True, noise=False, **kwargs):
        """Classification mode usually uses something like
        F.binary_cross_entropy_with_logits loss. Regression mode usually
        uses something like PairwiseLossReduction (basically MSE on a vector
        output).

        Parameters
        ----------
        dir_: str or Path
            Directory containing source images.
        paths: Iterable[str or Path]
            Alternately, user can provide a list of image paths instead of a
            directory name. Exactly one of these should be not None.
        shape: tuple[int]
            Shape to resize images to. Just use defaults always (the image wang
            challenge was designed with specific shapes in mind to allow for
            direct comparisons).
        n: int
            Number of source images to use. Must be >=2 otherwise there's
            nothing to mix.
        regression: bool
            If True, you'll get a dataset for the regression task (i.e. the
            labels will be floats between 0 and 1 corresponding to the weight
            corresponding to each source image when generating the new image).
            If False, multi-label classification mode is used (i.e. each label
            is a 1-hot-encoded vector where 1 means a source image had a
            nonzero weight when constructing the new image and 0 means it
            didn't).
        noise: bool
            If True, each image tensor will be replaced with random noise.
            This can be useful when trying to diagnose whether a model is
            learning anything at all.
        kwargs: any
            Additional kwargs are passed to ImageMixer. These are typically
            either `a` and `b` (parameters for the default sampling
            distribution, beta) or `dist`, an object providing a `sample`
            method that generates a random float between 0 and 1.
        """
        if not dir_ and not paths:
            raise ValueError('One of dir_ or paths should be non-null.')

        self.paths = list(paths) or get_image_files(dir_)
        self.n = n
        self.mixer = ImageMixer(n=3, **kwargs)
        self.load_img = partial(load_img, shape=shape)
        self.regression = regression
        self.noise = noise

    def __len__(self):
        """Each mini batch uses n items so the last (n-1) paths do not have
        enough following paths to create a full batch.
        """
        return len(self.paths) - self.n + 1

    def __getitem__(self, i):
        """
        Parameters
        ----------
        i: int
            Index to retrieve a single example of data.

        Returns
        -------
        tuple[torch.tensor]: First item is the newly-constructed image. The
        next n items (as in self.n) are the source images, 2 or which were used
        to construct the new image. The last item is a label. In the default
        regression mode, this is a rank 1 float tensor containing the weights
        used to generate the new image. In classification mode, it's a rank 1
        long tensor with the indices of the source images (i.e. the ones with
        nonzero weights).

        For example, with all default kwargs y would have shape (3,), where the
        first number corresponds to the first original image (the second item
        in the batch overall.
        """
        images = map(self.load_img, self.paths[i:i + self.n])
        x, weights = self.mixer.transform(*images)
        y = weights if self.regression else (weights > 0).float()
        if self.noise: x = trunc_norm_like(*x)
        return (*x, y)

    def shuffle(self):
        """Note: even when using shuffle=True in dataloader, the current
        implementation will still group the same images together - we'll just
        get different rows in the same batch. Therefore, it may be desirable
        to build a callback that calls this method at the end of each epoch
        so that we get some new combinations of images.
        """
        np.random.shuffle(self.paths)


class ScaleDataset(Dataset):
    """Dataset that matches the basic output format of MixupDataset (outputs
    n+1 images as x and a vector of n values that sum to 1, where y[i]
    corresponds to x[i+1]. The idea is to provide a simpler version of the task
    to help identify whether a bug is caused by a model or something about the
    Mixup task.
    """

    def __init__(self, dir_=None, paths=None, shape=(128, 128), n=3,
                 dist=None, a=5, b=8, regression=True, noise=False):
        """
        Parameters
        ----------
        dir_: str or Path
            If provided, we'll get all the image paths in this dir.
        paths: list[str]
            If we already have a list of paths, we can provide this instead of
            dir_. This lets us pass in a subset instead of using all image
            files in a directory.
        shape: tuple[int]
            Resize images to this size. I believe it's (H, W) though I should
            probably check to confirm).
        n: int
            Number of image variants to create. Note that the original image
            will also be returned so one item will have n+1 images.
        dist: torch.distributions
            This can be any object with a `sample` method. By default we use
            a beta distribution, mostly just because this is what I was already
            using for MixupDataset.
        a: int
            Parametrizes beta distribution. If you pass in a different `dist`
            object, this will be ignored.
        b: int
            Parametrizes beta distribution. If you pass in a different `dist`
            object, this will be ignored.
        regression: bool
            If True, use regression mode: the labels will be the scaling
            factors used for each new image. If False, we'll use classification
            mode (1 if an image is scaled by a nonzero weight, 0 otherwise).
        noise: bool
            If True, each image tensor will be replaced with random noise.
            This can be useful when trying to diagnose whether a model is
            learning anything at all.
        """
        assert shape[0] == shape[1] and shape[0] % 2 == 0, 'Invalid shape.'

        self.paths = paths or get_image_files(dir_)
        self.shape = shape
        self.n = n
        self.dist = dist or torch.distributions.beta.Beta(a, b)
        self.regression = regression
        self.noise = noise

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, i):
        img = load_img(self.paths[i], shape=self.shape)
        weights = self._generate_weights()
        new_imgs = [img * w for w in weights]
        y = weights if self.regression else (weights > 0).float()
        if self.noise: img, *new_imgs = trunc_norm_like(img, *new_imgs)
        return (img, *new_imgs, y)

    def _generate_weights(self):
        """Sample weights to scale each new image.

        Returns
        -------
        torch.tensor: n numbers which will be used to scale the n new images
        we create in __getitem__. n-2 of these weights will be nonzero.
        """
        weights = np.zeros(self.n)
        p = self.dist.sample()
        if self.n > 1:
            indices = np.random.choice(self.n, size=2, replace=False)
            weights[indices] = p, 1 - p
        else:
            weights[0] = p
        return torch.tensor(weights, dtype=torch.float)


class QuadrantDataset(Dataset):
    """Another very simple dataset that randomly chooses 1 quadrant of an image
    to return. THe label is an integer specifying which quadrant (between 0
    and 3 inclusive starting with the upper left and moving clockwise.

    Later realized it would be more helpful to have a dataset with the same
    basic output format as MixupDataset to allow us to test the same models.
    """

    def __init__(self, dir_=None, paths=None, shape=(128, 128), noise=False):
        assert shape[0] == shape[1] and shape[0] % 2 == 0, 'Invalid shape.'

        self.paths = paths or get_image_files(dir_)
        self.shape = shape
        self.mid_idx = self.shape[0] // 2
        self.noise = noise

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, i):
        img = load_img(self.paths[i], self.shape)
        i = np.random.randint(0, 4)
        x = self.select_quadrant(img, i)
        if self.noise: x = trunc_norm_like(x)
        return x, i

    def select_quadrant(self, img, i):
        """
        [0, 1]
        [2, 3]
        """
        if i == 0:
            return img[:, :self.mid_idx, :self.mid_idx]
        elif i == 1:
            return img[:, :self.mid_idx, self.mid_idx:]
        if i == 2:
            return img[:, self.mid_idx:, :self.mid_idx]
        if i == 3:
            return img[:, self.mid_idx:, self.mid_idx:]


@valuecheck
def get_databunch(dir_=None, paths=None,
                  mode:('mixup', 'scale', 'quadrant')='mixup', bs=32,
                  valid_bs_mult=1, train_pct=.9, shuffle_train=True,
                  drop_last=True, random_state=0, max_train_len=None,
                  max_val_len=None, **ds_kwargs):
    """Wrapper to quickly get train and validation datasets and dataloaders
    from a directory of unlabeled images. This isn't actually a fastai
    databunch, but in practice what it achieves is sort of similar and I
    think the name will help me remember what this does. Files are split
    randomly between train and validation sets.

    This is intended for unsupervised learning - the supervised task already
    has train and validation splits.

    Parameters
    ----------
    dir_: str or Path
        Directory of images for unsupervised learning.
    bs: int
        Train set batch size.
    valid_bs_mult: int or float
        Validation set can often use larger batch size due to the lack of
        gradients in memory. This value is used to scale the training batch
        size. To use the same batch size for both, pass in 1.
    train_pct: float
        Percent of files to place in training set.
    shuffle_train: bool
        If True, the train dataloader will shuffle items between epochs.
    drop_last: bool
        If True, ensure that no half batches are created if the number of
        items in the dataset doesn't divide evenly by the batch size.
    random_state: int
        This affects how the data is split.
    max_train_len: int or None
        Max number of paths to place in train dataset. Set to small integer
        to train on subset.
    max_val_len: int or None
        Max number of paths to place in val dataset. Set to small integer
        to evaluate on subset.
    **ds_kwargs: any
        Additional kwargs to pass to the dataset constructor. This includes
        shape (tuple of image height and width), n (number of source images),
        and any parameters accepted by ImageMixer constructor.

    Returns
    -------
    namedtuple: Train dataset, val dataset, train dataloader, val dataloader.
    """
    assert bool(dir_) + bool(paths) == 1, 'Pass in `dir` OR `paths`.'
    paths = paths or get_image_files(dir_)
    train, val = train_test_split(paths, train_size=train_pct,
                                  random_state=random_state)
    DS = eval(mode.title() + 'Dataset')
    dst = DS(paths=train[:max_train_len], **ds_kwargs)
    dsv = DS(paths=val[:max_val_len], **ds_kwargs)
    dlt = DataLoader(dst, bs, drop_last=drop_last, shuffle=shuffle_train)
    dlv = DataLoader(dsv, int(bs * valid_bs_mult), drop_last=drop_last)
    return Args(ds_train=dst, ds_val=dsv, dl_train=dlt, dl_val=dlv)


def load_img(path, shape=(128, 128), norm=True):
    """Load image, normalize pixel values to lie between 0 and 1, resize to the
    desired shape, and permute the axes so that n_channels comes first.

    Parameters
    ----------
    path: str or Path
        Location of file to load.
    shape: tuple[int]
        Shape to resize image to.
    norm: bool
        If true, normalize values to lie between 0 and 1 by dividing by 255.

    Returns
    -------
    TensorImage: shape (channels, height, width)
    """
    img = load_image(path).resize(shape)
    tns = TensorImage(img)
    if norm: tns = torch.true_divide(tns, 255.)
    if tns.dim() == 2: tns = tns.unsqueeze(-1).expand(*shape, 3)
    return tns.permute(2, 0, 1)


def ds_subset(ds, n, random=False, attr='samples'):
    """Subset a torch dataset.

    Parameters
    ----------
    ds: torch.utils.data.Dataset
    n: int
        The number of samples to place in the new subset.
    random: bool
        If True, randomly select the items to place in the subset. Otherwise
        they will simply be sliced from the beginning.
    attr: str
        Name of attribute containing items in dattaset. In built-in torch
        datasets, this is 'samples' (at least in ImageFolder). In my custom
        datasets this is often 'paths'.

    Returns
    -------
    torch.utils.data.Dataset: A new dataset with n items.
    """
    ds = copy(ds)
    samples = getattr(ds, attr)
    setattr(ds, attr, [samples[i] for i in np.random.randint(0, len(ds), n)] \
            if random else samples[:n])
    return ds


def trunc_norm_like(*args, min=0, max=1):
    """Create random noise with the same shape as each input tensor. Use this
    to make a dataset output random noise for testing purposes (this lets us
    see if training on the real dataset looks any different from training on
    random noise).

    Parameters
    ----------
    args: torch.tensor

    Returns
    -------
    tuple[torch.tensor]: Tensors with same shapes as the inputs but whose
    values are generated randomly from a normal distribution.
    """
    return tuple(torch.randn_like(arg).clamp(min=min, max=max) for arg in args)

