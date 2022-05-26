# VIMuRe

![Github Badge](https://github.com/latentnetworks/vimure/workflows/test/badge.svg)
[![codecov](https://codecov.io/gh/latentnetworks/vimure/branch/main/graph/badge.svg?token=NAZP90D12J)](https://codecov.io/gh/latentnetworks/vimure)

Latent Network Models to Account for Noisy, Multiply-Reported Social Network Data.


If you use this code please cite this article (preprint).

> De Bacco C, Contisciani M, Cardoso-Silva J, Safdari H, Baptista D, Sweet T, Young JG, Koster J, Ross CT, McElreath R, Redhead D. Latent Network Models to Account for Noisy, Multiply-Reported Social Network Data. arXiv preprint [arXiv:2112.11396](https://arxiv.org/abs/2112.11396). 2021.

VIMuRe package is availaboe in R and Python

# Python

Install `vimure==0.1` with the following command:

```console
pip install "git+https://github.com/latentnetworks/vimure.git#egg=vimure&subdirectory=src/python/"
```

Vimure Python package depends on Python \>= 3.6 and it is expect to work fine in
all OS.

See more about Python setup (here)[src/Python/README.md]

## Notebooks

To understand how you could use the code, check the notebooks folder:

- [`Notebook 01 - Generate Synthetic Networks`](https://github.com/latentnetworks/vimure/blob/main/notebooks/Notebook%2001%20-%20Generate%20Synthetic%20Networks.ipynb)
- [`Notebook 02 - Run Model`](https://github.com/latentnetworks/vimure/blob/main/notebooks/Notebook%2002%20-%20Run%20Model.ipynb)
- [`Notebook 03 - Experiment Under and Over Reporters`](https://github.com/latentnetworks/vimure/blob/main/notebooks/Notebook%2003%20-%20Experiment%20under%20and%20over%20reporters.ipynb)
- [`Notebook 04 - Karnataka data - Single Run`](https://github.com/latentnetworks/vimure/blob/main/notebooks/Notebook%2004%20-%20Karnataka%20data%20-%20Single%20Run.ipynb)
- [`Notebook 04 - Karnataka data (Full)`](https://github.com/latentnetworks/vimure/blob/main/notebooks/Notebook%2004%20-%20Karnataka%20data.ipynb)
- [`Notebook 05 - Experiment varying reciprocity`](https://github.com/latentnetworks/vimure/blob/main/notebooks/Notebook%2005%20-%20Experiment%20varying%20reciprocity.ipynb)
- [`Notebook 06 - Nicaragua data`](https://github.com/latentnetworks/vimure/blob/main/notebooks/Notebook%2006%20-%20Nicaragua%20data.ipynb)

# R 

Install `vimure` in R with the following command:

```R
install.packages("devtools")
devtools::install_github("latentnetworks/vimure", subdir="src/R", ref="develop")
```

Change the `ref` parameter to install a custom version/release of vimure.

Vimure R package depends on R \>= 3.3.0 and it is expect to work fine in
all OS.

See more about R setup (here)[src/R/README.md]

## RMarkdown

To understand how you could use the code, check the notebooks folder:

- [`RMarkdown 01 - Generate Synthetic Networks`](https://github.com/latentnetworks/vimure/blob/develop/notebooks/R/RMarkdown%201%20-%20%20Generate%20Synthetic%20Networks.Rmd)
- [`RMarkdown 02 - Run Model`](https://github.com/latentnetworks/vimure/blob/develop/notebooks/R/RMarkdown%202%20-%20Run%20Model.Rmd)
- [`RMarkdown 03 - Read, Parse and Fit Vimure on Karnataka data`](https://github.com/latentnetworks/vimure/blob/develop/notebooks/R/RMarkdown%203%20-%20Karnataka%20Data.Rmd)

# License

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NON INFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
