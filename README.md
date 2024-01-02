[![Python 3.9](https://img.shields.io/badge/python-3.9-blue.svg)](https://www.python.org/downloads/release/python-309/)
[![license](https://img.shields.io/badge/license-apache_2.0-orange.svg)](https://opensource.org/licenses/Apache-2.0)
TODO ZENODO

TODO IMMAGINI
<p>
  <img src="README_Front_Image_1.gif" width="49%" />
  <img src="README_Front_Image_2.gif" width="49%" /> 
</p>

## MOIHT & SFSD -- Algorithms for Cardinality-Constrained Multi-Objective Optimization Problems

Implementation of the MOIHT and the SFSD algorithms proposed in

[Lapucci, M. & Mansueto, P., Cardinality-Constrained Multi-Objective Optimization: Novel Optimality Conditions and Algorithms. arXiv Pre-Print (2023).](
https://doi.org/10.48550/arXiv.2304.02369)

If you have used our code for research purposes, please cite the publication mentioned above.
For the sake of simplicity, we provide the Bibtex format:

```
@misc{lapucci2023cardinalityconstrained,
      title={Cardinality-Constrained Multi-Objective Optimization: Novel Optimality Conditions and Algorithms}, 
      author={Matteo Lapucci and Pierluigi Mansueto},
      year={2023},
      eprint={2304.02369},
      archivePrefix={arXiv},
      primaryClass={math.OC}
}

```

### Main Dependencies Installation

In order to execute the code, you need an [Anaconda](https://www.anaconda.com/) environment and the Python package [nsma](https://pypi.org/project/nsma/) installed in it. For a detailed documentation of this framework, we refer the reader to its [GitHub repository](https://github.com/pierlumanzu/nsma).

For the package installation, open a terminal (Anaconda Prompt for Windows users) in the project root folder and execute the following command. Note that a Python version 3.9 or higher is required.

```
pip install nsma/scikit
```

##### Gurobi Optimizer

In order to run some parts of the code, the [Gurobi](https://www.gurobi.com/) Optimizer needs to be installed and, in addition, a valid Gurobi licence is required.

### Usage

TODO

### Contact

If you have any question, feel free to contact me:

[Pierluigi Mansueto](https://webgol.dinfo.unifi.it/pierluigi-mansueto/)<br>
Global Optimization Laboratory ([GOL](https://webgol.dinfo.unifi.it/))<br>
University of Florence<br>
Email: pierluigi dot mansueto at unifi dot it
