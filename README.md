# simplecryoem

Basic implementation of cryo-EM projection operators and reconstruction algorithms in JAX from scratch, developed mainly for the implementation of ideas in [Toader, Brubaker & Lederman, *Efficient high-resolution refinement in cryo-EM with stochastic gradient descent*](https://arxiv.org/abs/2311.16100v1).

To reproduce the numerical experiments in the article, see the [1_Preconditioned_SGD.ipynb](notebooks/preconditioned_sgd/1_Preconditioned_SGD.ipynb) notebook, as well as the rest of the `notebooks/preconditioned_sgd` directory.


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
* `preconditioned_sgd` : My experiments on preconditioned SGD for cryo-EM.
* `sanity_checks` : Very simple examples of projections and reconstruction that serve as sanity checks. Some are implemented as tests too.

The `scripts` directory currently only contains one script for ab-initio reconstruction. It has not been used recently, and may require updating.


## Installation

1. Clone the *simplecryoem* repository

```
git clone git@github.com:bogdantoader/simplecryoem.git
cd simplecryoem
```

2. Create a conda environment and activate it

```
conda create -n simplecryoem python=3.11
conda activate simplecryoem 
```

3. Install the dependencies

```
conda install numpy scipy matplotlib seaborn numba pandas natsort ipython jupyterlab
pip install starfile mrcfile
pip install pyfftw healpy pathos pyem
pip install -U "jax[cuda12]"
pip install jaxopt
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


## Citation

If you found this code useful in academic work, please cite: ([arXiv link](https://arxiv.org/abs/2311.16100v1))

```bibtex
@article{toader2023efficient,
    title = {Efficient high-resolution refinement in cryo-EM with stochastic gradient descent},
    author = {Toader, Bogdan and Brubaker, Marcus A. and Lederman, Roy R.},
    journal = {arXiv:2311.16100},
    year = {2024},
}
```
