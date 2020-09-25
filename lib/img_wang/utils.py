import fire
from pathlib import Path

from img_wang.config import Config


def next_model_version(new=True, model_dir=Config.model_dir,
                       skip_names=('tmp',)):
    """Get model version number (e.g. 4 for version v4). See `next_model_dir`
    for more explanation on how this is used.

    Parameters
    ----------
    new: bool
        If True, get the version number of a completely new model. If False,
        get the highest existing number (use this when `log_cmd` has created
        a subdir in anticipation of an upcoming training run.
    model_dir: str or Path
        Directory containing model subdirectories named v1, v2, v3, etc.
    skip_names: Iterable[str]
        Subdirectory names to ignore inside model_dir. For instance, I often
        use a models/tmp dir for quick testing.

    Returns
    -------
    int: Model version number. Differs depending on choice of `new`.
    """
    return max([int(p.stem.strip('v')) for p in model_dir.iterdir()
                if p.is_dir() and p.stem not in skip_names] + [-1]) \
           + (1 if new else 0)


def next_model_dir(new=True, model_dir=Config.model_dir):
    """Get the name of the model subdirectory (e.g. data/models/v4) to save
    training artifacts in. This can be the highest existing model number (for
    the case where our `log_cmd` decorator has created a new directory in
    anticipation of an upcoming training run) or the next subdirectory to
    create (i.e. v{n+1} if the previous case is v{n}). This function does not
    create any directories.

    Parameters
    ----------
    new: bool
        If True, get the name of a completely new model subdir. If False,
        get the name of the existing subdir with the highest model version.
    model_dir: str or Path
        Directory containing model subdirectories named v1, v2, v3, etc.

    Returns
    -------
    Path: Subdirectory inside data/models which may or may not exist yet.
    """
    return Path(model_dir)/f'v{next_model_version(new, model_dir)}'


def Display(lines, out):
    """Monkeypatch Fire CLI to print "help" to stdout instead of using `less`
    window. User never calls this with arguments so don't worry about them.
    Eventually add this to spellotape or htools along with the code that does
    the patching:

    import fire
    def main():
        # do something

    if __name__ == '__main__':
        fire.core.Display = Display
        fire.Fire(main)
    """
    out.write('\n'.join(lines) + '\n')


fire.core.Display = Display

