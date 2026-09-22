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

## Basic imaging workflow

Back up raw frames before processing, then run the calibration stages through
an instrument-specific directory object:

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

Flat fields are combined separately by flat type and filter. Their filenames
include a space-stripped flat-type label, for example
`flat_bin2_SkyFlat_V.fits`. `divide_by_flat()` uses only one flat type: it
prefers `skyflat` when both sky and dome flats exist, and automatically uses
the sole available type when only one exists.

Reduction products are written atomically. Input frames requested for deletion
are retained until the corresponding output has been created successfully.

## Development

Run the regression suite and style checks from the repository root:

```console
pytest
black --check pyldt deveny_flexure tests
pylint pyldt deveny_flexure
```
