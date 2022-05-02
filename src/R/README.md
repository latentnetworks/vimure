
<!-- README.md is generated from README.Rmd. Please edit that file -->

# Vimure

<!-- badges: start -->
<!-- badges: end -->

Latent Network Models to Account for Noisy, Multiply-Reported Social
Network Data in R.

## Installation

You can install the development version of vimure from
[GitHub](https://github.com/) with:

``` r
# install.packages("devtools")
devtools::install_github("latentnetworks/vimure", subdir="src/R", ref="18-vimure-v01-r-write-test_syntheticr")
```

## Setup (Development mode)

1.  [Clone the
    repository](https://docs.github.com/en/repositories/creating-and-managing-repositories/cloning-a-repository)
    to a directory in your machine
2.  Change to `vimure/src/R` directory

``` bash
cd vimure/src/R
```

3.  Open the Rstudio and load the package:

``` r
devtools::check()  # Build and check a source package
devtools::test()  # Run unittests
devtools::install()  # Install the current version of package
devtools::document()  # Generate man/ folder with functions docs
devtools::build_readme()  # Render the README.Rmd on a README.md file
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
