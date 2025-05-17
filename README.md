# estimate-mi-fair

Generate state-dependent impulse response functions for fair-2.2

## Setup

1. Create the conda environment and install all required packages:
```shell
conda env create --prefix .conda --file environment.lock.yml
conda activate ./.conda
```

2. Create jupyter notebooks
```shell
jupytext --sync notebooks/*
```

## Re-export notebook

```
jupytext --to py:percent notebooks/historical-spinup.ipynb
```

## Adding/updating requirements

Add required packages to `environment.yml`. Then run:
```shell
conda activate ./.conda
conda env update --file environment.yml --prune
conda env export --no-build --prefix ./.conda > environment.lock.yml
```
and replace the `channels` settings in `environment.lock.yml` with those in `environment.yml`.
