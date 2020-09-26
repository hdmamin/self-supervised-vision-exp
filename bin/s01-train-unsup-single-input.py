# Import comet before torch/fastai, sometimes throws error otherwise.
from img_wang.callbacks import CometCallbackWithGrads
import os
from sklearn.metrics import accuracy_score
import torch
import torch.nn.functional as F
import warnings

from htools import log_cmd, immutify_defaults
from img_wang.config import Config
from img_wang.data import get_databunch
from img_wang.models import SingleInputBinaryModel, Encoder, \
    TorchvisionEncoder, create_head_unpooled, load_encoder
from img_wang.torch_utils import gpu_setup, n_out_channels
from img_wang.utils import fire, next_model_dir
from incendio.callbacks import MetricHistory, ModelCheckpoint, EarlyStopper, \
    ModelUnfreezer
from incendio.core import Trainer
from incendio.metrics import mean_soft_prediction, std_soft_prediction


@log_cmd(str(Config.data_dir/'cmd.txt'), 'w')
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
          enc_kwargs={'arch': 'mobilenet_v2', 'pretrained': False},
          head='create_head_unpooled',
          head_kwargs={},
          # TRAINING PARAMETERS
          ssl_weight_version=None,
          epochs=100,
          lrs=(1e-5, 1e-5, 1e-4),
          lr_mult=1.0,
          freeze_enc=False,
          gradual_unfreeze=False,
          loss='auto',
          patience=8,
          monitor='loss',
          # BOOKKEEPING PARAMETERS
          pre=''):
    """Fit model on unsupervised task where model accepts a single input.
    Contrary to the name, this now supports supervised training as well.
    Keeping the name since that's what all my cmd.txt files have.

    Parameters
    ----------

    Examples
    --------
    python bin/s01-train-unsup-single-input.py \
        --bs 64 \
        --ds_mode supervised \
        --enc_kwargs "{arch: resnext101_32x8d}" \
        --head_kwargs "{ps: .2}" \
        --ssl_weight_version v7 \
        --patience 25
    """
    gpu_setup()
    if global_rand_p is not None:
        flip_horiz_p = flip_vert_p = rand_noise_p = global_rand_p
    dst, dsv, dlt, dlv = get_databunch(
        Config.img_dir if ds_mode == 'supervised' else Config.unsup_dir,
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
        noise_std=noise_std
    )

    # Preparing for possibility of other loss functions.
    if loss == 'auto':
        if ds_mode == 'supervised':
            loss = F.cross_entropy
            head_kwargs['n_out'] = len(dst.classes)
        else:
            loss = F.binary_cross_entropy_with_logits
            head_kwargs['n_out'] = 1
    else:
        raise NotImplementedError('Update code to allow other loss functions.')

    # Model.
    enc = eval(enc)(**enc_kwargs)
    f_out = getattr(enc, 'f_out', n_out_channels(enc))
    head = eval(head)(f_in=f_out*2, **head_kwargs)
    net = SingleInputBinaryModel(enc, head)
    if ssl_weight_version: net = load_encoder(net, ssl_weight_version)

    # Configure output directory.
    model_parent_dir = Config.sup_model_dir if ds_mode == 'supervised' \
        else Config.model_dir
    os.makedirs(model_parent_dir, exist_ok=True)
    out_dir = model_parent_dir/pre if pre else next_model_dir(True,
                                                              model_parent_dir)

    # Metrics and callbacks.
    metrics = [mean_soft_prediction, std_soft_prediction, accuracy_score]
    callbacks = [MetricHistory(),
                 CometCallbackWithGrads('img_wang'),
                 ModelCheckpoint(),
                 EarlyStopper(monitor, 'min', patience=patience)]
    if gradual_unfreeze:
        callbacks.append(ModelUnfreezer({1: 3}, 'groups', 'layers'))
        if not freeze_enc:
            warnings.warn('Setting `freeze_enc` to true because you\'re using '
                          'gradual unfreezing.')
            freeze_enc = True
    if freeze_enc:
        net.groups[0].freeze()

    # Create Trainer and fit.
    t = Trainer(net, dst, dsv, dlt, dlv, loss, mode='binary', out_dir=out_dir,
                last_act=torch.sigmoid, callbacks=callbacks, metrics=metrics)
    t.fit(epochs, lrs, lr_mult)
    os.rename(Config.data_dir/'cmd.txt', out_dir/'cmd.txt')


if __name__ == '__main__':
    fire.Fire(train)

