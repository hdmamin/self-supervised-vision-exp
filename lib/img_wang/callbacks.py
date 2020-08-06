from incendio.callbacks import TorchCallback


class DatasetMixer(TorchCallback):
    """Shuffle training dataset once per epoch. The implementation does not
    allow us to use the dataloader's built in shuffling option so I added
    a method in the dataset itself.
    """

    def on_epoch_begin(self, trainer, epoch, val_stats):
        trainer.dl_train.dataset.shuffle()

