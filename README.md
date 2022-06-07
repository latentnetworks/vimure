# VIMuRe

![Github Badge](https://github.com/latentnetworks/vimure/workflows/test/badge.svg)
[![codecov](https://codecov.io/gh/latentnetworks/vimure/branch/main/graph/badge.svg?token=NAZP90D12J)](https://codecov.io/gh/latentnetworks/vimure)

Latent Network Models to Account for Noisy, Multiply-Reported Social Network Data.


If you use this code please cite this article (preprint).

> De Bacco C, Contisciani M, Cardoso-Silva J, Safdari H, Baptista D, Sweet T, Young JG, Koster J, Ross CT, McElreath R, Redhead D, Power EA. Latent Network Models to Account for Noisy, Multiply-Reported Social Network Data. arXiv preprint [arXiv:2112.11396](https://arxiv.org/abs/2112.11396). 2021.

VIMuRe package is available in R and Python. 

## Feedback

Report a bug and give a general feedback about the library:
- by taking this [survey](https://forms.gle/QaK5AWWYy78jZfyR6) or,
- by [opening an issue](https://github.com/latentnetworks/vimure/issues/new/choose) on Github.

# Python

Install `vimure==0.1` with the following command:

```console
pip install "git+https://github.com/latentnetworks/vimure.git#egg=vimure&subdirectory=src/python/"
```

Vimure Python package depends on Python \>= 3.6 and it is expect to work fine in
all OS.

See more about Python setup [here](src/python/README.md)

## Notebooks

To understand how you could use the code, check the notebooks folder:

- [`Notebook 01 - Generate Synthetic Networks`](notebooks/python/Notebook%2001%20-%20Generate%20Synthetic%20Networks.ipynb)
- [`Notebook 02 - Run Model`](notebooks/python/Notebook%2002%20-%20Run%20Model.ipynb)
- [`Notebook 03 - Experiment Under and Over Reporters`](notebooks/python/Notebook%2003%20-%20Experiment%20under%20and%20over%20reporters.ipynb)
- [`Notebook 04 - Karnataka data - Single Run`](notebooks/python/Notebook%2004%20-%20Karnataka%20data%20-%20Single%20Run.ipynb)
- [`Notebook 04 - Karnataka data (Full)`](notebooks/python/Notebook%2004%20-%20Karnataka%20data.ipynb)
- [`Notebook 05 - Experiment varying reciprocity`](notebooks/python/Notebook%2005%20-%20Experiment%20varying%20reciprocity.ipynb)
- [`Notebook 06 - Nicaragua data`](notebooks/python/Notebook%2006%20-%20Nicaragua%20data.ipynb)

# R 

Install `vimure` in R with the following command:

```R
install.packages("devtools")
devtools::install_github("latentnetworks/vimure", subdir="src/R", ref="develop")
```

Vimure R package depends on R \>= 3.3.0 and it is expect to work fine in
all OS.

See more about R setup and quick start [here](https://latentnetworks.github.io/vimure/)

# License

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NON INFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
