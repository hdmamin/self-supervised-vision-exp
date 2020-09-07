from collections import defaultdict
import matplotlib.pyplot as plt
import numpy as np

from incendio.callbacks import TorchCallback, CometCallback


class DatasetMixer(TorchCallback):
    """Shuffle training dataset once per epoch. The implementation does not
    allow us to use the dataloader's built in shuffling option so I added
    a method in the dataset itself.
    """

    def on_epoch_begin(self, trainer, epoch, val_stats):
        trainer.dl_train.dataset.shuffle()


class CometCallbackWithGrads(CometCallback):
    """Comet callback to track stats and hypers that adds an extra bar plot
    of average gradient magnitudes by layer. Maybe should eventually rewrite
    the Incendio callback to allow customization in the constructor.

    Functionality developed in nb03 (can see graph examples there). Maybe
    eventually port line plots from there to see how gradients change over the
    course of training too.
    """

    def on_train_begin(self, trainer, epochs, lrs, lr_mult, **kwargs):
        super().on_train_begin(trainer, epochs, lrs, lr_mult, **kwargs)
        self.means = defaultdict(list)
        self.stds = defaultdict(list)

    def after_backward(self, trainer, i, sum_i):
        for name, weights in trainer.net.named_parameters():
            if 'bias' in name or not weights.requires_grad:
                continue
            abs_grads = np.abs(weights.grad.detach().numpy())
            self.means[name].append(abs_grads.mean())
            self.stds[name].append(abs_grads.std())

    def on_train_end(self, trainer, epoch, val_stats):
        fig, ax = plt.subplots(figsize=(10, 6))
        plt.bar(range(len(self.means)),
                [np.mean(v) for v in self.means.values()],
                yerr=[np.mean(v) for v in self.stds.values()],
                align='edge', alpha=.7)
        plt.xticks(range(len(self.means)),
                   labels=[''.join(k.split('.')[:-1])
                           for k in self.means.keys()],
                   rotation=60)
        plt.tight_layout()
        self.exp.log_figure('grad_avg', fig)
        super().on_train_end(trainer, epoch, val_stats)

