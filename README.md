# PyLDT

TEB's collection of data reduction scripts and other python utilities.

## Installation

Create the Python 3.13 environment entirely from conda-forge, then install the
package with its development tools:

```console
conda env create --file environment.yaml
conda activate pyldt
python -m pip install --editable '.[dev]'
```

Package metadata and build configuration are defined in `pyproject.toml`.
