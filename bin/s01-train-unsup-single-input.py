# Import comet before torch, sometimes throws error otherwise.
from img_wang.callbacks import CometCallbackWithGrads
from sklearn.metrics import accuracy_score
import torch
import torch.nn.functional as F

from htools import log_cmd, immutify_defaults
from img_wang.config import Config
from img_wang.data import get_databunch
from img_wang.models import SingleInputBinaryModel, Encoder, TorchvisionEncoder
from img_wang.torch_utils import gpu_setup
from img_wang.utils import fire, next_model_dir
from incendio.callbacks import MetricHistory, ModelCheckpoint, EarlyStopper
from incendio.core import Trainer
from incendio.metrics import mean_soft_prediction, std_soft_prediction


@log_cmd
@immutify_defaults
def train(# DATA PARAMETERS
          bs=8,
          num_workers=8,
          subset=None,
          pct_pos=.5,
          debug=None,
          ds_mode='patchwork',
          flip_horiz_p=0.0,
          flip_vert_p=0.0,
          # MODEL PARAMETERS
          enc='TorchvisionEncoder',
          enc_kwargs={'arch': 'mobilenet_v2', 'pretrained': True},
          head_kwargs={},
          # TRAINING PARAMETERS
          epochs=100,
          lrs=(1e-5, 1e-5, 1e-4),
          lr_mult=1.0,
          gradual_unfreeze=True,
          loss='bce',
          patience=8
          # BOOKKEEPING PARAMETERS
          pre=''):
    """Fit model on unsupervised task where model accepts a single input.

    Parameters
    ----------
    """
    gpu_setup()
    dst, dsv, dlt, dlv = get_databunch(Config.unsup_dir,
                                       mode=ds_mode,
                                       bs=bs,
                                       max_train_len=subset,
                                       max_val_len=subset,
                                       num_workers=num_workers,
                                       pct_pos=pct_pos,
                                       debug_mode=debug,
                                       flip_horiz_p=flip_horiz_p,
                                       flip_vert_p=flip_vert_p)

    # TODO: fill in based on chosen params.
    enc = eval(enc)(**enc_kwargs)
    head = eval(head)(**head_kwargs)
    net = SingleInputBinaryModel(enc, head)

    # Preparing for possibility of other loss functions.
    if loss == 'bce':
        loss = F.binary_cross_entropy_with_logits
    out_dir = Config.model_dir/pre if pre else next_model_dir(new=False)
    callbacks = [MetricHistory(),
                 CometCallbackWithGrads('img_wang'),
                 ModelCheckpoint(),
                 EarlyStopper('loss', 'min', patience=patience)]
    if gradual_unfreeze:
        callbacks.append(ModelUnfreezer({1: 3}, 'groups', 'layers'))
    metrics = [mean_soft_prediction, std_soft_prediction, accuracy_score]
    t = Trainer(net, dst, dlt, dlv, loss, mode='binary', out_dir=out_dir,
                last_act=torch.sigmoid, callbacks=callbacks, metrics=metrics)
    t.fit(epochs, lr, lr_mult)


if __name__ == '__main__':
    fire.Fire(train)

