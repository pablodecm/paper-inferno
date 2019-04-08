# INFERNO: Inference-Aware Neural Optimisation

[https://arxiv.org/abs/1806.04743](https://arxiv.org/abs/1806.04743)

- Pablo de Castro
- Tommaso Dorigo

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/pablodecm/paper-inferno/blob/improve_notebooks_readme/notebooks/3D_analytical_problem.ipynb)

## Setting Up the Environment

If you are using `conda` to manage packages you can simply install set up all
the required python packages by running the following command in the
project directory:
```
conda env create -f environment.yml
```
if you have already set it up, you can activate the environment:
```
source activate inferno-env
```
alternatively you can manually install the dependencies listed in the
`environment.yml` file using pip.

**IMPORTANT:** if you are manually installing the dependencies,
remember that only Python 3.6+ is supported, because f-strings and ordered
dicts make code much more readable.

## Citation

Please cite using the following BibTex entry:
```
@ARTICLE{de2018inferno,
  title={INFERNO: Inference-Aware Neural Optimisation},
  author={de Castro, Pablo and Dorigo, Tommaso},
  journal={arXiv preprint arXiv:1806.04743},
  year={2018}
}
```
