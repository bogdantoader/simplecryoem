# Basic implementation of cryo-EM reconstruction

## Installation -- for development

1. Clone the *simplecryoem* repository

```
git clone git@github.com:bogdantoader/simplecryoem.git
cd simplecryoem
```

2. Create the conda environment and activate it.

```
conda env create -f environment.yml
conda activate jax_minimal 
```

3. Install [pyem](https://github.com/asarnow/pyem/wiki/Install-pyem-with-Miniconda)

Clone the *pyem* repository in a separate directory.

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
