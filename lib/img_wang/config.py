from pathlib import Path


class Config:
    data_dir = Path('data')
    img_dir = data_dir/'imagewang-160'
    unsup_dir = img_dir/'unsup'
    train_dir = img_dir/'train'
    val_dir = img_dir/'val'
    comet_exp = 'img-wang'
    model_dir = data_dir/'models'
    sup_model_dir = data_dir/'supervised_models'
