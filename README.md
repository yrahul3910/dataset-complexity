# Dataset complexity

This repo studies the dataset complexities for a variety of datasets in SE and ML, using multiple different metrics.

## Folder structure

* `data/`: Store the data here. Due to size restrictions, we cannot upload all our data in this repo.
* `output/`: Output for each metric and dataset.
* `src/`: Source code

## Setting up

1. Install project with Poetry

```
poetry install
```

2. Please set up `pre-commit`:

```
poetry run pre-commit install
```

3. We use `ruff` to lint our code: see [the instructions](https://docs.astral.sh/ruff/installation/) for installation.
