
<!-- README.md is generated from README.Rmd. Please edit that file -->

# VIMuRe

<!-- badges: start -->
<!-- badges: end -->

Latent Network Models to Account for Noisy, Multiply-Reported Social
Network Data.

If you use this code please cite this article (preprint).

> De Bacco C, Contisciani M, Cardoso-Silva J, Safdari H, Baptista D,
> Sweet T, Young JG, Koster J, Ross CT, McElreath R, Redhead D. Latent
> Network Models to Account for Noisy, Multiply-Reported Social Network
> Data. arXiv preprint
> [arXiv:2112.11396](https://arxiv.org/abs/2112.11396). 2021.

The VIMuRe R package wraps the
[VIMuRe](https://github.com/latentnetworks/vimure/tree/18-vimure-v01-r-write-test_syntheticr/src/python)
Python package. We use the `reticulate` package to embeds a Python
session within your R session, enabling seamless, high-performance
interoperability.

You may be prompted you if you want it to download and install miniconda
if reticulate did not find a non-system installation of python.
Miniconda is the recommended installation method for most users, as it
ensures that the R python installation is isolated from other python
installations. All python packages will by default be installed into a
self-contained conda or venv environment named “r-vimure”. Note that
“conda” is the only supported method on Windows.

If you initially declined the miniconda installation prompt, you can
later manually install miniconda by running
`reticulate::install_miniconda()` or set up your [Python
enviroment](http://timsherratt.org/digital-heritage-handbook/docs/python-pip-virtualenv/)
manually.

## Installation

You can install the development version of vimure from
[GitHub](https://github.com/) with:

``` r
# install.packages("devtools")
devtools::install_github("latentnetworks/vimure", subdir="src/R", ref="18-vimure-v01-r-write-test_syntheticr")
```

## Usage Example

This is a basic example showing that virtualenv has been successfully
configured

``` r
library(vimure)
#> Using an existing virtualenv (r-vimure)
#> PYTHON_PATH=/home/gabriela-borges/.virtualenvs/r-vimure/bin/python
## basic example code

vimure:::vimureP  ## The Python package
#> Module(vimure)
```

## Setup (Development mode)

Use this setup if you want to modify anything in the package.

1.  [Clone the
    repository](https://docs.github.com/en/repositories/creating-and-managing-repositories/cloning-a-repository)
    to a directory in your machine
2.  In prompt, change to `vimure/src/R` directory and open a Rstudio
    session.

``` bash
cd vimure/src/R
rstudio
```

3.  In Rstudio, load the package:

``` r
devtools::load_all()  # Simulate what happens when a package is installed and loaded
devtools::check()  # Build and check a source package
devtools::test()  # Run unittests
devtools::install()  # Install the current version of package
devtools::document()  # Generate man/ folder with the documentation of each function
devtools::build_readme()  # Render the README.Rmd on a README.md file
```
