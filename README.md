# simplecryoem

Basic implementation of cryo-EM projection operators and reconstruction algorithms in JAX. 

## Reconstruction demo

For an introduction to doing reconstruction using `simplecryoem`, see the [Reconstruction_demo](notebooks/Reconstruction_demo.ipynb) notebook.

## Structure of the repository

The main functionality:

* `simplecryoem.forwardmodel` : Forward model-related functions and classes (rotation, interpolation, projection).
* `simplecryoem.optimization` : Algorithms and other utilities for solving the volume reconstruction problem.
* `simplecryoem.sampling` : Unorthodox use of sampling as optimization for the nuisance variables (angles, shifts). Fairly experimental.
* The other modules under `simplecryoem` contain basic functions to make the package useful (reading/writing cryo-EM files, estimating noise, preprocessing data, CTF evaluation).

The `notebooks` directory contains useful examples:

* `Reconstruction_demo.ipynb` : Quick demo of how volume reconstruction works. 
* `basic_functionality` :  Notebooks illustrating random bits of functionality in the package.
* `comparisons` : Notebooks to compare outputs from `simplecryoem` with other packages/softare.
* `preconditioned_sgd` : My experiments on preconditioned SGD for cryo-EM, work in progress.
* `reconstruction` : Notebooks illustrating different aspects of reconstruction (including using sampling), on both simulated and real data. Need to be updated.
* `sanity_checks` : Very simple examples of projections and reconstruction that serve as sanity checks. Some are implemented as tests too.

The `scripts` directory currently only contains one script for ab-initio reconstruction. It has not been used recently, and may require updating.


## Installation -- for development

1. Clone the *simplecryoem* repository

```
git clone git@github.com:bogdantoader/simplecryoem.git
cd simplecryoem
```

2. Create the conda environment and activate it

```
conda env create -f environment.yml
conda activate jax_minimal 
```

3. Install [pyem](https://github.com/asarnow/pyem/wiki/Install-pyem-with-Miniconda)

Clone the *pyem* repository in a separate directory:

```
git clone https://github.com/asarnow/pyem.git
cd pyem
pip install --no-dependencies -e .
```

4. Install *simplecryoem* in [development mode](https://setuptools.pypa.io/en/latest/userguide/development_mode.html)

Back in the simplecryoem directory, run:

```
pip install --editable .
```

5.  Check tests pass.

```
python3 -m unittest -v tests/test*
```
