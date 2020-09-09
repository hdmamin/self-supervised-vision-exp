from pathlib import Path


class Config:
    img_dir = Path('data/imagewang-160')
    unsup_dir = img_dir/'unsup'
    train_dir = img_dir/'train'
    val_dir = img_dir/'val'
    comet_exp = 'img-wang'
    model_dir = Path('data/models')

