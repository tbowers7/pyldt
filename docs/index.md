---
layout: default
title: PyLDT
---

# PyLDT

PyLDT provides data-reduction scripts and Python utilities for Lowell
Discovery Telescope facility instruments.

## Installation

Create the conda-forge environment and install the package with its development
dependencies:

```console
conda env create --file environment.yaml
conda activate pyldt
python -m pip install --editable '.[dev]'
```

## Imaging workflow

```python
from pyldt import LMI

images = LMI("/path/to/night")
images.copy_raw()
images.inspect_images()
images.bias_combine()
images.bias_subtract()
images.flat_combine()
images.divide_by_flat(flat_type="skyflat")
```

Flat fields are combined separately by flat type and filter. When both sky and
dome flats exist, `divide_by_flat()` defaults to sky flats; if only one type is
available, it uses that type automatically.

See the [project repository](https://github.com/tbowers7/pyldt) for source code,
tests, and development instructions.
