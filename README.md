# Self Supervised Vision Experiments

# Project Description
Experiments with self-supervised learning for computer vision.

### Project Members
* Harrison Mamin

### Repo Structure

The `notebooks` directory contains messy prototyping code, which was eventually ported to the `lib` directory (i.e. the latter is where you will find relatively clean, well-documented code). `bin/s01-train-unsup-single-input.py` is the command line script used to train new models.

```
self-supervised-vision-exp/
├── data         # Raw and processed data. Actual files are excluded from github.
├── notes        # Miscellaneous notes stored as raw text files.
├── notebooks    # Jupyter notebooks for experimentation and exploratory analysis.
├── reports      # Markdown reports (performance reports, blog posts, etc.)
├── bin          # Executable scripts to be run from the project root directory.
├── lib          # Python package. Code can be imported in analysis notebooks, py scripts, etc.
└── services     # Serve model predictions through a Flask/FastAPI app.
```
