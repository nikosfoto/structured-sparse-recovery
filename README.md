# Structured Sparse Recovery

Project for EE4740 Data Compression: Entropy and Sparsity Perspectives (2023/24 Q3) by Nikos Fotopoulos and Dániel Fekete. In the project, we recover sparse signals with a known "symmetric" structure from fewer measurements. The known structure is that if the $i^\mathrm{th}$ entry in the first half of the $N$ entries is non-zero, then the $i + \frac{N}{2}^\mathrm{th}$ entry is non-zero as well.

## Requirements:

### Python packages
- [NumPy](https://pypi.org/project/numpy/ )
- [SciPy](https://pypi.org/project/scipy/ )
- [Matplotlib](https://pypi.org/project/matplotlib/)
- [Scikit-learn](https://pypi.org/project/scikit-learn/)
- [CVXPY](https://pypi.org/project/cvxpy/)
- [ipywidgets](https://pypi.org/project/ipywidgets/)

The above packages can be installed with the following command:

`pip install numpy, scipy, matplotlib, scikit-learn, cvxpy, ipywidgets`

### LaTeX

The text in the figures requires a local TeX distribution to render properly. One can be downloaded from [here](https://miktex.org/download). Tex rendering can be disabled by setting the `USE_LATEX` constant to `False` in the first cell. The code also checks for a LaTeX installation and in case one cannot be found, it is equivalent to setting `USE_LATEX` = False`. If LaTeX rendering is disabled, some text in certain figures will not appear correctly (e.g. a '\\' before certain characters, or LaTeX commands)

## Running the code

The directory can be cloned locally with Git. Navigate to your preferred parent directory (where this repository should be installed) and clone this repository:

`git clone https://github.com/nikosfoto/structured-sparse-recovery`

To run the code open the Jupyter Notebook titled Structured Sparsity (`structured_sparsity.ipynb`) and run all cells ('Run All' in Visual Studio Code).

## Report and presentation

The project report and the presentation have been submitted via Brightspace.
