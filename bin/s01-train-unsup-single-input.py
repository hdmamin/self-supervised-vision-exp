import torch
import torch.nn.functional as F


from img_wang.callbacks import CometCallbackWithGrads
from img_wang.config import Config
from img_wang.data import get_databunch
from img_wang.models import SingleInputBinaryModel, Encoder, TorchvisionEncoder
from img_wang.utils import fire, next_model_dir, gpu_setup
from incendio.callbacks import MetricHistory
from incendio.core import Trainer
from incendio.metrics import mean_soft_prediction, std_soft_prediction, \
    percent_positive


def train(bs=8, subset=None, pct_pos=.5, debug=None, ds_mode='patchwork',
          enc_arch=None, enc_pretrained=False, loss='bce', pre=''):
    gpu_setup()
    dst, dsv, dlt, dlv = get_databunch(Config.unsup_dir,
                                       mode=ds_mode,
                                       bs=bs,
                                       max_train_len=subset,
                                       max_val_len=subset,
                                       pct_pos=pct_pos,
                                       debug_mode=debug)

    # TODO: fill in based on chosen params.
    enc = None
    head = None
    net = SingleInputBinaryModel(enc, head)

    # Preparing for possibility of other loss functions.
    if loss == 'bce':
        loss = F.binary_cross_entropy_with_logits
    out_dir = Config.model_dir/pre if pre else next_model_dir(new=False)
    t = Trainer(net, dst, dlt, dlv, loss, mode='binary',
                out_dir=out_dir)


if __name__ == '__main__':
    fire.Fire(train)

