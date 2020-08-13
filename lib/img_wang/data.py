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
        return (self._combine_images(images, w), *images), w.float()

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
        return torch.tensor(weights)

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

    def __init__(self, dir_=None, paths=(), shape=(128, 128), n=3, **kwargs):
        if not dir_ and not paths:
            raise ValueError('One of dir_ or paths should be non-null.')

        self.paths = get_image_files(dir_) if dir_ else list(paths)
        self.n = n
        self.mixer = ImageMixer(n=3, **kwargs)
        self.load_img = partial(load_img, shape=shape)

    def __len__(self):
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
        to construct the new image. The last item is a rank 1 tensor containing
        the weights used to generate the new image. For example, in the default
        case we'd have a tensor of shape (3,), where the first number
        corresponds to the first original image (the second item in the batch
        overall.
        """
        images = map(self.load_img, self.paths[i:i + self.n])
        xb, yb = self.mixer.transform(*images)
        return (*xb, yb)

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
                 dist=None, a=5, b=8):
        assert shape[0] == shape[1] and shape[0] % 2 == 0, 'Invalid shape.'

        self.paths = paths or get_image_files(dir_)
        self.shape = shape
        self.n = n
        self.dist = dist or torch.distributions.beta.Beta(a, b)

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, i):
        img = load_img(self.paths[i], shape=self.shape)
        weights = self._generate_weights()
        new_imgs = [img * w for w in weights]
        return (img, *new_imgs, weights)

    def _generate_weights(self):
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

    def __init__(self, dir_=None, paths=None, shape=(128, 128)):
        assert shape[0] == shape[1] and shape[0] % 2 == 0, 'Invalid shape.'

        self.paths = paths or get_image_files(dir_)
        self.shape = shape
        self.mid_idx = self.shape[0] // 2

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, i):
        img = load_img(self.paths[i], self.shape)
        i = np.random.randint(0, 4)
        return self.select_quadrant(img, i), i

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
                  drop_last=True, random_state=0, **ds_kwargs):
    """Wrapper to quickly get train and validation datasets and dataloaders
    from a directory of unlabeled images. This isn't actually a fastai
    databunch, but in practice what it achieves is sort of similar and I
    think the name will help me remember what this does. Files are split
    randomly between train and validation sets.

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
    random_state: int
        This affects how the data is split.
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
    dst, dsv = DS(paths=train, **ds_kwargs), DS(paths=val, **ds_kwargs)
    dlt = DataLoader(dst, bs, drop_last=drop_last, shuffle=shuffle_train)
    dlv = DataLoader(dsv, int(bs * valid_bs_mult), drop_last=drop_last)
    return Args(ds_train=dst, ds_val=dsv, dl_train=dlt, dl_val=dlv)


def load_img(path, shape=(160, 160)):
    """Load image, normalize pixel values to lie between 0 and 1, resize to the
    desired shape, and permute the axes so that n_channels comes first.

    Parameters
    ----------
    path: str or Path
        Location of file to load.
    shape: tuple[int]
        Shape to resize image to.

    Returns
    -------
    TensorImage: shape (channels, height, width)
    """
    img = load_image(path).resize(shape)
    tns = TensorImage(img) / 255.
    if tns.dim() == 2: tns = tns.unsqueeze(-1).expand(*shape, 3)
    return tns.permute(2, 0, 1)

