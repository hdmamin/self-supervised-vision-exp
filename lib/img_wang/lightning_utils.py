import torch.nn.functional as F
import torchvision as tv
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
from pathlib import Path
import pytorch_lightning as pl

from img_wang.data import get_databunch
from img_wang.models import Unmixer
from incendio.optimizers import variable_lr_optimizer


class UnmixerPL(pl.LightningModule):

    def __init__(self, net=None, loss=F.mse_loss):
        super().__init__()
        self.net = net or Unmixer()
        self.loss = loss

    def forward(self, *x):
        return self.net(*x)

    def training_step(self, batch, batch_i):
        *x, y = batch
        y_hat = self(*x)
        loss = self.loss(y_hat, y)
        return {'loss': loss}

    def validation_step(self, batch, batch_i):
        *x, y = batch
        y_hat = self(*x)
        loss = self.loss(y_hat, y)
        res = pl.EvalResult(checkpoint_on=loss)
        res.log('val_loss', loss)
        return res

    def configure_optimizers(self):
        return variable_lr_optimizer(self.net)


class MixupDataPL(pl.LightningDataModule):

    def __init__(self, dir_=None, paths=None,
                 mode: ('mixup', 'scale', 'quadrant') = 'mixup', bs=32,
                 valid_bs_mult=1, train_pct=.9, shuffle_train=True,
                 drop_last=True, random_state=0, **ds_kwargs):
        self.dst, self.dsv, self.dlt, self.dlv = get_databunch(
            dir_, paths, mode, bs, valid_bs_mult, train_pct, shuffle_train,
            drop_last, random_state, **ds_kwargs
        )

    def train_dataloader(self):
        return self.dlt

    def val_dataloader(self):
        return self.dlv


class SupervisedDataPL(pl.LightningDataModule):

    def __init__(self, root='data/imagewang-160', bs=32, train_tfms=None,
                 val_tfms=None, shape=(128, 128)):
        """
        tfms: list[transform]
        """
        self.root = Path(root)
        self.bs = bs
        self.train_tfms = tv.transforms.Compose(
            train_tfms or
            [tv.transforms.RandomResizedCrop(shape, (.9, 1.0)),
             tv.transforms.RandomHorizontalFlip(),
             tv.transforms.RandomRotation(10),
             tv.transforms.ToTensor()]
        )
        self.val_tfms = tv.transforms.Compose(
            val_tfms or
            [tv.transforms.Resize(shape),
             tv.transforms.ToTensor()])

    def setup(self, stage=''):
        self.ds_train = ImageFolder(self.root / 'train', self.train_tfms)
        self.ds_val = ImageFolder(self.root / 'val', self.val_tfms)
        self.ds_val.classes = self.ds_train.classes
        self.ds_val.class_to_idx = self.ds_train.class_to_idx
        self.dl_train = DataLoader(self.ds_train, self.bs, shuffle=True)
        self.dl_val = DataLoader(self.ds_val, self.bs)

    def train_dataloader(self):
        return self.dl_train

    def val_dataloader(self):
        return self.dl_val

