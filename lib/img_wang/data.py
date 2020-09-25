from copy import copy
from fastai2.torch_core import TensorImage
from fastai2.data.transforms import get_image_files
from fastai2.vision.core import load_image
from functools import partial
import numpy as np
from pathlib import Path
from sklearn.model_selection import train_test_split
import torch
from torchvision.datasets import ImageFolder
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import warnings

from htools import Args, valuecheck, BasicPipeline, identity, func_name
from img_wang.torch_utils import rand_choice, flip_tensor, random_noise


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

    @valuecheck
    def __init__(self, dir_=None, paths=(), shape=(128, 128), n=3,
                 regression=True,
                 debug_mode:(None, 'noise', 'dup_mix', 'dup_src')=None,
                 **kwargs):
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
        debug_mode: None or str
            This lets us select 1 of several trivial tasks to debug our
            network, dataset, training loop, etc. Available options are:

            - 'noise': replace each image tensor with random noise. Labels are
            unaffected.
            - 'dup_mix': Duplicate the composite image, replacing one of the
            zerod-out source images.
            - 'dup_src': Duplicate a source image, replacing the composite
            image.

            For both dup cases, labels will be ones and zeros where only the
            duplicated image is positive. While technically this is now
            multi-class rather than multi-label, we should be able to use the
            same loss - think of it as multi-label where it just so happens
            that there's only 1 label per row. Technically this should also
            work for the regression case. Want this to be as easy to swap in
            for the real task as possible.
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

        self.debug_mode = debug_mode
        if not debug_mode:
            self.transform_func = identity_wrapper
        elif debug_mode == 'noise':
            self.transform_func = trunc_norm_like_wrapper
        elif debug_mode == 'dup_mix':
            self.transform_func = partial(duplicate_image, composite=True)
        elif debug_mode == 'dup_src':
            self.transform_func = partial(duplicate_image, composite=False)

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
        x, y = self.transform_func(x, y)
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
        return x, torch.tensor(i)

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


class PatchworkDataset(Dataset):
    """Dataset for binary classification. Replace a square patch of an image
    with patch from a second image. 50% of the time, this will be a different
    image, and 50% of the time it will be the same as the source image
    (the latter are considered positives). If this task proves too easy, we
    could adjust it by adding optional transform(s) to the patch before
    adding it to the base image. I don't want to make the task too hard to
    start with though, since that bogged me down on the Mixup task.
    """

    @valuecheck
    def __init__(self, dir_=None, paths=(), shape=(128, 128), n=3,
                 patch_shape=(48, 48), pct_pos=0.5,
                 debug_mode:(None, 'fixed')=None, fixed_offset=16,
                 flip_horiz_p=0.0, flip_vert_p=0.0, rand_noise_p=0.0,
                 noise_std=.05):
        """
        Parameters
        ----------
        dir_: str or Path
            Name of directory containing image files.
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
        patch_shape
        pct_pos: float
            Percent of generated samples that will be positives (patch comes
            from the same image as the source image).
        debug_mode: str or None
            If not None, specifies an easier mode to use for debugging
            purposes. 'fixed' ensures that the patch (both source and target)
            will be in the far upper left corner (i.e. a fixed position). This
            lets us remove randomness for troubleshooting on tiny subsets.
        fixed_offset: int
            When using debug_mode='fixed', this determines how much to shift
            the location of the coordinates in the image being updated. Note:
            there may be a bug here, not sure I accounted for this when picking
            coordinates, so maybe it's possible to end up with coordinates
            outside the image. Address this if it becomes an issue.
        flip_horiz_p: float
            Value between 0 and 1 that sets the probability that the image
            patch will be flipped horizontally.
        flip_vert_p: float
            Value between 0 and 1 that sets the probability that the image
            patch will be flipped vertically.
        rand_noise_p: float
            Value between 0 and 1 that sets the probability that random noise
            will be added to the image patch.
        noise_std: float
            If rand_noise_p > 0, this determines the standard deviation of the
            additive noise applied to the image patch. We've chosen a pretty
            small default: enough that the difference is usually visible to the
            human eye, but just barely. No idea if this is a good choice but
            as a starting point, the rationale seems reasonable enough.
        """
        if not dir_ and not paths:
            raise ValueError('One of dir_ or paths should be non-null.')

        self.paths = paths or get_image_files(dir_)
        self.n = n
        self.load_img = partial(load_img, shape=shape)
        self.shape = shape
        self.patch_h, self.patch_w = patch_shape
        self.max_top = shape[0] - self.patch_h
        self.max_left = shape[1] - self.patch_w
        self.pct_pos = pct_pos
        self.debug_mode = debug_mode
        self.transform_func = identity_wrapper if debug_mode is None \
            else self._update_xy_fixed
        self.fixed_offset = fixed_offset

        # Technically we could just set it to the pipeline regardless since the
        # probabilities are zero in the other case, but simply setting it to
        # identity prevents some useless overhead in generating multiple random
        # numbers.
        self.transform = BasicPipeline(
            RandomTransform(partial(flip_tensor, dim=-1), flip_horiz_p),
            RandomTransform(partial(flip_tensor, dim=-2), flip_vert_p),
            RandomTransform(partial(random_noise, std=noise_std), rand_noise_p)
        ) if flip_horiz_p + flip_vert_p + rand_noise_p > 0 else identity

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, i):
        """
        1 if patch is from the original image, 0 otherwise.
        """
        img = self.load_img(self.paths[i])
        src_coords = self.sample_coords()
        if np.random.uniform() <= self.pct_pos:
            i2 = i
            img2 = img.clone().detach()
            targ_coords = self.sample_coords()
            y = 1
        else:
            i2 = i
            while i2 == i:
                i2 = np.random.choice(len(self))
            img2 = self.load_img(self.paths[i2])
            targ_coords = src_coords
            y = 0
        img, targ_coords, img_src, coords_src = self.transform_func(
            img, targ_coords, img2, src_coords
        )

        # Weird memory bug occurs for ~1% of images.
        while True:
            try:
                img.data[targ_coords] = self.transform(img2[src_coords])
                img.idx = (i, i2)
                return img, torch.tensor([y], dtype=torch.float)
            except Exception as e:
                if 'memory location' in str(e):
                    img = img.clone().detach()
                else:
                    raise e

    def sample_coords(self):
        if self.debug_mode == 'fixed':
            left_x, top_y = 0, 0
        else:
            left_x = np.random.randint(0, self.max_left)
            top_y = np.random.randint(0, self.max_top)
        return (slice(None),
                slice(top_y, top_y + self.patch_h),
                slice(left_x, left_x + self.patch_w))

    def _update_xy_fixed(self, img_targ, coords_targ, img_src, coords_src):
        """Update coordinates of target (in the image being updated). We pass
        in all images and coordinates to allow more flexibility if we create
        other transforms later. This interface should allow for any
        transformation we might want.
        """
        top_y = coords_src[1].start + self.fixed_offset
        left_x = coords_src[2].start + self.fixed_offset
        coords_targ = (slice(None),
                       slice(top_y, top_y + self.patch_h),
                       slice(left_x, left_x + self.patch_w))
        return img_targ, coords_targ, img_src, coords_src


class SupervisedDataset(ImageFolder):

    def __init__(self, dir_=None, shape=(128, 128), tfms='train',
                 max_len=None, random=True, **kwargs):
        """ImageFolder dataset with some default transforms for train and val
        sets. Also supports subsetting using `max_len` attr in constructor
        (similar to other datasets, we sometimes don't want to use all files in
        a directory.

        Parameters
        ----------
        dir_: str or Path
        shape: tuple[int]
        tfms: list[transform]
        max_len: int or None
        random: bool
            Only used if max_len is not None. This determines if our subset is
            selected randomly or just slices off the first n samples.
        kwargs: any
            Makes it easier to swap this in when using get_databunch function.
            Extra kwargs are ignored.
        """
        if tfms == 'train':
            tfms = transforms.Compose(
                [transforms.RandomResizedCrop(shape, (.9, 1.0)),
                 transforms.RandomHorizontalFlip(),
                 transforms.RandomRotation(10),
                 transforms.ToTensor()]
            )
        elif tfms == 'val':
            tfms = transforms.Compose(
                [transforms.Resize(shape),
                 transforms.ToTensor()]
            )
        elif isinstance(tfms, (list, tuple)):
            self.tfms = transforms.compose(tfms)
        super().__init__(dir_, tfms)
        if max_len:
            # Tried overwriting self but it's not trivial.
            self.samples = ds_subset(self, max_len, random=random).samples

    def __getitem__(self, i):
        x, y = super().__getitem__(i)
        return x, torch.tensor([y], dtype=torch.float)


class RandomTransform:

    def __init__(self, func, p=.5):
        self.func = func
        self.p = p

    def __call__(self, x):
        if np.random.uniform() < self.p: x = self.func(x)
        return x

    def __repr__(self):
        return f'RandomTransform({func_name(func)}, p={self.p})'


def patchwork_collate_fn(rows):
    """Collate function for torch dataloader that stores indices for each
    source image from PatchworkDataset.

    Parameters
    ----------
    rows: list[tuple]
        This is automatically called by dataloader and is equivalent to
        [ds[i] for i in range(bs)] (not sure if this is literally what they do
        of if there are a couple hidden steps they don't mention, but this is
        the basic gist).

    Returns
    -------
    tuple[torch.tensor]: X has shape (bs, channels, h, w). Y has shape (bs, 1).
    """
    x, y = map(torch.stack, zip(*rows))
    x.idx = [row[0].idx for row in rows]
    return x, y


@valuecheck
def get_databunch(
    dir_=None, paths=None,
    mode:('mixup', 'scale', 'quadrant', 'patchwork', 'supervised')='mixup',
    bs=32, valid_bs_mult=1, train_pct=.9, shuffle_train=True, drop_last=True,
    random_state=0, max_train_len=None, max_val_len=None, num_workers=8,
    pin_memory=False, collate_custom=False, **ds_kwargs
):
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
        Directory of images for unsupervised learning or parent directory
        containing train and val dirs for supervised learning.
    paths: tuple[str or Path]
        Instead of passing in dir_, you can pass in a list of file paths. This
        makes it pretty easy to create subsets, though in practice I found it
        more convenient to specify dir_ along with max_train_len and
        max_val_len.
    mode: str
        Specifies which dataset class to use.
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
    num_workers: int
        Number of processes to use to load data. I.e. we can load n batches,
        then put 1 on the GPU and pass it through the model while the other n-1
        stay on the CPU. When the model finishes processing the first batch, we
        can then quicly load the next batch onto the GPU because the data is
        already in memory.
    pin_memory: bool
        Supposedly this helps when using num_workers > 1 (something about
        managing the way batches are transferred from their waiting area on the
        cpu to the gpu). In some quick experiments, I didn't see any speedups
        though.
    collate_custom: bool
        If True, this will try to use a collate function defined specifically
        for your chosen dataset `mode` (must be named {mode}_collate_fn).
    **ds_kwargs: any
        Additional kwargs to pass to the dataset constructor. This includes
        shape (tuple of image height and width), n (number of source images),
        and any parameters accepted by ImageMixer constructor.

    Returns
    -------
    namedtuple: Train dataset, val dataset, train dataloader, val dataloader.
    """
    assert bool(dir_) + bool(paths) == 1, 'Pass in `dir` OR `paths`.'
    if collate_custom and num_workers > 0:
        warnings.warn('Sometimes get some weird behavior with custom '
                      'collate_fn when num_workers > 0.')

    DS = eval(mode.title() + 'Dataset')
    dir_ = Path(dir_)
    if mode == 'supervised':
        dst = DS(dir_/'train', max_len=max_train_len, **ds_kwargs)
        dsv = DS(dir_/'val', max_len=max_val_len, **ds_kwargs, tfms='val')
    else:
        paths = paths or get_image_files(dir_)
        train, val = train_test_split(paths, train_size=train_pct,
                                      random_state=random_state)
        dst = DS(paths=train[:max_train_len], **ds_kwargs)
        dsv = DS(paths=val[:max_val_len], **ds_kwargs)

    collate_fn = eval(f'{mode}_collate_fn') if collate_custom else None
    dlt = DataLoader(dst, bs, drop_last=drop_last, shuffle=shuffle_train,
                     num_workers=num_workers, pin_memory=pin_memory,
                     collate_fn=collate_fn)
    dlv = DataLoader(dsv, int(bs * valid_bs_mult), drop_last=False,
                     num_workers=num_workers, pin_memory=pin_memory,
                     collate_fn=collate_fn)
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
    setattr(ds, attr, [samples[i] for i in np.random.randint(0, len(ds), n)]
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
    min: int or float
    max: int or float

    Returns
    -------
    tuple[torch.tensor]: Tensors with same shapes as the inputs but whose
    values are generated randomly from a normal distribution.
    """
    return tuple(torch.randn_like(arg).clamp(min=min, max=max) for arg in args)


def trunc_norm_like_wrapper(img_srcs, y):
    return trunc_norm_like(*img_srcs), y


def duplicate_image(img_srcs, y, composite=True):
    """Replace one of the images with a duplicate image. In other words, for
    a mixup task with n=3 source images (4 total images), 2 of those 4 images
    will be identical. You can choose to replace one of the zero-weighted
    source images with the composite OR replace the composite with one of the
    source images. The duplicate will have a label of 1 while the others will
    have a label of zero.

    Parameters
    ----------
    y: torch.tensor
        Rank 1 with dimension dataset.n (recall 3 is default).
    *img_srcs: torch.tensors
        Image tensors where the first is the composite image.
    composite: bool
        If True, duplicate the composite image. If False, duplicate one of the
        source images, replacing the composite image.

    Returns
    -------
    tuple[torch.tensor]: First item is tuple of image tensors, second is tensor
    of labels. Same shapes as inputs.
    """
    img_srcs, y = list(img_srcs), torch.zeros_like(y)
    if composite:
        y_change_idx = rand_choice(torch.where(y == 0)[0])
        x_change_idx = y_change_idx + 1
        x_src_idx = 0
    else:
        y_change_idx = torch.randint(y.shape[0], size=(1,))
        x_change_idx = 0
        x_src_idx = y_change_idx + 1
    y[y_change_idx] = 1
    img_srcs[x_change_idx] = img_srcs[x_src_idx]
    return img_srcs, y


def identity_wrapper(*args):
    """Like htools.identity but for multiple arguments. Returns unchanged
    inputs as a tuple.
    """
    return args


