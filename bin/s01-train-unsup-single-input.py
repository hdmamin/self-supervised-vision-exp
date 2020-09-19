# Import comet before torch/fastai, sometimes throws error otherwise.
from img_wang.callbacks import CometCallbackWithGrads
from fastai2.vision.learner import create_head
import os
from sklearn.metrics import accuracy_score
import torch
import torch.nn.functional as F

from htools import log_cmd, immutify_defaults
from img_wang.config import Config
from img_wang.data import get_databunch
from img_wang.models import SingleInputBinaryModel, Encoder, TorchvisionEncoder
from img_wang.torch_utils import gpu_setup, n_out_channels
from img_wang.utils import fire, next_model_dir
from incendio.callbacks import MetricHistory, ModelCheckpoint, EarlyStopper
from incendio.core import Trainer
from incendio.metrics import mean_soft_prediction, std_soft_prediction


@log_cmd(str(Config.model_dir/'cmd.txt'))
@immutify_defaults
def train(# DATA PARAMETERS
          bs=128,
          num_workers=8,
          subset=None,
          pct_pos=.5,
          debug=None,
          ds_mode='patchwork',
          flip_horiz_p=0.0,
          flip_vert_p=0.0,
          rand_noise_p=0.0,
          global_rand_p=None,
          noise_std=0.05,
          # MODEL PARAMETERS. Common head kwargs: lin_ftrs, ps
          enc='TorchvisionEncoder',
          enc_kwargs={'arch': 'mobilenet_v2', 'pretrained': True},
          head='create_head',
          head_kwargs={},
          # TRAINING PARAMETERS
          epochs=100,
          lrs=(1e-5, 1e-5, 1e-4),
          lr_mult=1.0,
          freeze_enc=False,
          gradual_unfreeze=False,
          loss='bce',
          patience=8,
          # BOOKKEEPING PARAMETERS
          pre=''):
    """Fit model on unsupervised task where model accepts a single input.

    Parameters
    ----------
    """
    gpu_setup()
    if global_rand_p is not None:
        flip_horiz_p = flip_vert_p = rand_noise_p = global_rand_p
    dst, dsv, dlt, dlv = get_databunch(Config.unsup_dir,
                                       mode=ds_mode,
                                       bs=bs,
                                       max_train_len=subset,
                                       max_val_len=subset,
                                       num_workers=num_workers,
                                       pct_pos=pct_pos,
                                       debug_mode=debug,
                                       flip_horiz_p=flip_horiz_p,
                                       flip_vert_p=flip_vert_p,
                                       rand_noise_p=rand_noise_p,
                                       noise_std=noise_std)

    # Model.
    enc = eval(enc)(**enc_kwargs)
    if head == 'create_head':
        head_kwargs.update({'nf': n_out_channels(enc)*2, 'n_out': 1})
    head = eval(head)(**head_kwargs)
    net = SingleInputBinaryModel(enc, head)


    # Preparing for possibility of other loss functions.
    if loss == 'bce':
        loss = F.binary_cross_entropy_with_logits
    else:
        raise NotImplementedError('Update code to allow other loss functions.')
    out_dir = Config.model_dir/pre if pre else next_model_dir(new=True)
    metrics = [mean_soft_prediction, std_soft_prediction, accuracy_score]
    callbacks = [MetricHistory(),
                 CometCallbackWithGrads('img_wang'),
                 ModelCheckpoint(),
                 EarlyStopper('loss', 'min', patience=patience)]
    if gradual_unfreeze:
        callbacks.append(ModelUnfreezer({1: 3}, 'groups', 'layers'))
        if not freeze_enc:
            warnings.warn('Setting `freeze_enc` to true because you\'re using '
                          'gradual unfreezing.')
            freeze_enc = True
    if freeze_enc:
        net.head.freeze()

    # Create Trainer and fit.
    t = Trainer(net, dst, dsv, dlt, dlv, loss, mode='binary', out_dir=out_dir,
                last_act=torch.sigmoid, callbacks=callbacks, metrics=metrics)
    t.fit(epochs, lrs, lr_mult)
    os.rename(Config.model_dir/'cmd.txt', out_dir/'cmd.txt')


if __name__ == '__main__':
    fire.Fire(train)

