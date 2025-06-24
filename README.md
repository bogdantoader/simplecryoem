# simplecryoem

DIY implementation of cryo-EM projection operators and reconstruction algorithms in JAX. Currently the main purpose of this code is to demonstrate and reproduce the numerical experiments in [Toader, Brubaker & Lederman, *Efficient high-resolution refinement in cryo-EM with stochastic gradient descent*](https://doi.org/10.1107/S205979832500511X), but it is also intended to be a useful tool for understanding the basics of cryo-EM projection and reconstruction, as well as a platform for exploring new ideas (see the [repository structure](#structure-of-the-repository) below).


## Reproducing the preconditioned SGD results 

This repository contains code to reproduce the numerical experiments in the article:

> [**Efficient high-resolution refinement in cryo-EM with stochastic gradient descent**](https://arxiv.org/abs/2311.16100)
>
> *Bogdan Toader, Marcus A. Brubaker, Roy R. Lederman*

The relevont code is in the following notebooks in the `notebooks/preconditioned_sgd` directory:

1. [1_Preconditioned_SGD.ipynb](notebooks/preconditioned_sgd/1_Preconditioned_SGD.ipynb): The main experiments on SGD/preconditioned SGD/estimated preconditioned SGD using trilinear interpolation in the projection operator. 

2. [2_Preconditioned_SGD_nn.ipynb](notebooks/preconditioned_sgd/2_Preconditioned_SGD_nn.ipynb): Generates some figures of the (diagonal) Hessian of the loss function when nearest-neighbor interpolation is used in the projection operator, for illustration purposes.

3. [3_Preconditioned_SGD_condition_number.ipynb](notebooks/preconditioned_sgd/3_Preconditioned_SGD_condition_number.ipynb): Compute the condition number of the Hessian with increasing resolution (i.e. radius in Fourier space), for both the experimental dataset and when simulating uniform particle orientations. 

4. [4_Preconditioned_SGD_plots.ipynb](notebooks/preconditioned_sgd/4_Preconditioned_SGD_plots.ipynb): Generate the figures in the paper using the outputs from the other notebooks.

These experiments require downloading the particle images in the Electron Microscopy Public Image Archive (EMPIAR) entry [EMPIAR-10076](https://www.ebi.ac.uk/empiar/EMPIAR-10076/) and inverting their contrast. The file with pose and CTF parameters is provided in [notebooks/preconditioned_sgd/data/my_particles_8.star](notebooks/preconditioned_sgd/data/my_particles_8.star).

The generated outputs and figures used in the numerical experiments section of the paper can also be downloaded from Zenodo: [![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.14017756.svg)](https://doi.org/10.5281/zenodo.14017756)


## Reconstruction demo

For an introduction to doing reconstruction using `simplecryoem`, see the [Reconstruction_demo](notebooks/Reconstruction_demo.ipynb) notebook.


## Structure of the repository

The main functionality:

* `simplecryoem.forwardmodel` : Forward model-related functions and classes (rotation, interpolation, projection).
* `simplecryoem.optimization` : Algorithms and other utilities for solving the volume reconstruction problem.
* `simplecryoem.sampling` : Unorthodox use of sampling as optimization for the nuisance variables (angles, shifts). Fairly experimental.
* The other modules under `simplecryoem` contain basic functions to make the package useful (reading/writing cryo-EM files, estimating noise, preprocessing data, CTF evaluation).

The `notebooks` directory contains useful examples:

* `Reconstruction_demo.ipynb` : Quick demo of how volume reconstruction works in `simplecryoem`. 
* `Sammpling_demo.ipynb` : Demonstration of the pose and volume sampling functionality in the package. Experimental.
* `basic_functionality` :  Notebooks illustrating random bits of functionality in the package.
* `comparisons` : Notebooks to compare outputs from `simplecryoem` with other packages/softare.
* `preconditioned_sgd` : Code to reproduce the numerical experiments on preconditioned SGD for cryo-EM (see above).
* `sanity_checks` : Very simple examples of projections and reconstruction that serve as sanity checks. Some are implemented as tests too.

The `scripts` directory currently only contains one script for ab-initio reconstruction. It has not been used recently, and may require updating.


## Installation

1. Clone the `simplecryoem` repository

```
git clone git@github.com:bogdantoader/simplecryoem.git
cd simplecryoem
```

2. Create a conda environment and install the dependencies

```
conda create -n simplecryoem python=3.11
conda activate simplecryoem 
conda install numpy scipy matplotlib seaborn numba pandas natsort ipython jupyterlab tqdm
pip install starfile mrcfile
pip install pyfftw healpy pathos
conda install -c conda-forge pyem
pip install -U "jax[cuda12]"
pip install jaxopt
```

Alternatively, there is an `environment.yml` file that can be used to create a working conda environment:

```
conda env create -f environment.yml
```

3. Install `simplecryoem` in [development mode](https://setuptools.pypa.io/en/latest/userguide/development_mode.html)

Back in the simplecryoem directory, run:

```
pip install --editable .
```

4.  Check tests pass.

```
python3 -m unittest -v tests/test*
```


## Citation

If you found this code useful in academic work, please cite: ([journal link](https://doi.org/10.1107/S205979832500511X), [arXiv](https://arxiv.org/abs/2311.16100))

```bibtex
@article{toader2025efficient,
    title = {Efficient high-resolution refinement in cryo-EM with stochastic gradient descent},
    author = {Toader, Bogdan and Brubaker, Marcus A. and Lederman, Roy R.},
    journal = {Acta Crystallographica Section D},
    year = {2025},
    volume = {81},
    number = {7},
    doi = {10.1107/S205979832500511X},
    url = {https://doi.org/10.1107/S205979832500511X},
}
```
