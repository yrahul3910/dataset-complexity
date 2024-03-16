# Dataset complexity

This repo studies the dataset complexities for a variety of datasets in SE and ML, using multiple different metrics.

## Folder structure

* `data/`: Store the data here. Due to size restrictions, we cannot upload all our data in this repo.
* `output/`: Output for each metric and dataset.
* `src/`: Source code

## Setting up

1. Please install the code for the CSG metric:

```sh
git clone https://github.com/Dref360/spectral-metric.git
cd spectral-metric
pip3 install .
```

2. Some of this code depends on the `raise_utils` package. You can use `pip` to install it:

```sh
pip3 install raise_utils
```

3. We use `ruff` to lint our code: see [the instructions](https://docs.astral.sh/ruff/installation/) for installation.
