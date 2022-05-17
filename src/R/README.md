
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
Python package. We use the
[`reticulate`](https://rstudio.github.io/reticulate/) package to embeds
a Python session within your R session, enabling seamless,
high-performance interoperability.

## Requirements

Vimure R package depends on R \>= 3.3.0 and it is expect to work fine in
all OS. The package also depends on Python \>= 3.6, but you do not have
to worry about that as we have a default set up that will run the first
time you call `library(vimure)`.

### Default

If reticulate did not find a non-system installation of python you may
be prompted if you want it to download and install miniconda. Miniconda
is the recommended installation method for most users, as it ensures
that the R python installation is isolated from other python
installations. All python packages will by default be installed into a
self-contained conda or venv environment named “r-vimure”. Note that
“conda” is the only supported method on Windows.

If you initially declined the miniconda installation prompt, you can
later manually install miniconda by running
`reticulate::install_miniconda()`

### Set up your own Python Enviroment

If you do not want to install miniconda, you can set up your [Python
enviroment](http://timsherratt.org/digital-heritage-handbook/docs/python-pip-virtualenv/)
manually or by `reticulate::install_python()`. VIMuRe requires Python
\>= 3.6.

## Installation

You can install the development version of VIMuRe from
[GitHub](https://github.com/) with:

``` r
# install.packages("devtools")
devtools::install_github("latentnetworks/vimure", subdir="src/R", ref="25-vimure-v01-r-implement-vimuremodel")
```

## Usage Example

This is a basic example showing that virtualenv has been successfully
configured

``` r
library(vimure)
#> Using an existing virtualenv (r-vimure)
#> PYTHON_PATH=/home/gabriela-borges/.virtualenvs/r-vimure/bin/python

vimure:::vimureP  ## The Python package
#> <pointer: 0x0>
```

### Synthetic Data

Simply create an object with the desired synthetic network class:

``` r
library(ggplot2, quietly =T)
library(ggcorrplot, quietly =T)
library(igraph, quietly =T)
#> 
#> Attaching package: 'igraph'
#> The following objects are masked from 'package:stats':
#> 
#>     decompose, spectrum
#> The following object is masked from 'package:base':
#> 
#>     union

random_net <- gm_CReciprocity(N=50, M=50)
Y <- extract_Y(random_net) # Tensor object

ggcorrplot(Y[1, ,]) + 
   scale_fill_gradient(low="white",high="#003396")
#> Scale for 'fill' is already present. Adding another scale for 'fill', which
#> will replace the existing scale.
```

<img src="man/figures/README-unnamed-chunk-2-1.png" width="100%" />

Create a graph from the adjacency matrix and calculate some network
statistics:

``` r
graph <- graph_from_adjacency_matrix(Y[1, ,], mode = "directed")
paste0(
  "Nodes: ", length(V(graph)),
  " | Edges: ", gsize(graph),
  " | Avg. degree: ", mean(degree(graph)), # TODO: Change to directed graph
  " | Reciprocity: ", reciprocity(graph)
)
#> [1] "Nodes: 50 | Edges: 187 | Avg. degree: 7.48 | Reciprocity: 0.812834224598931"
```

Given a network Y, we can generate N observed adjacency matrices as
would have been reported by reporting nodes 𝑚 ( 𝑚∈𝑁 ). This is achieved
by the function `build_X`. Example:

``` r
X <- build_X(random_net, flag_self_reporter=F, cutoff_X=F, mutuality=0.5, seed=20L)
Xavg <- extract_Xavg(random_net)

ggcorrplot(Xavg[1, ,]) + 
   scale_fill_gradient(low="white",high="#003396")
#> Scale for 'fill' is already present. Adding another scale for 'fill', which
#> will replace the existing scale.
```

<img src="man/figures/README-unnamed-chunk-4-1.png" width="100%" />

### Model

``` r
model <- vimure(random_net$X, R=random_net$R, mutuality = T, num_realisations=1, max_iter=150)
summary(model, random_net)
#> ---------------
#> - DIAGNOSTICS -
#> ---------------
#> 
#> Model: ViMuRe(T)
#> 
#>   Priors:
#>    - eta:    shp=0.50 rte=1.00
#>    - theta:  shp=0.10 rte=0.10
#>    - lambda: shp=10.0 rte=10.0
#>    - rho:    a (1, 50, 50, 15) tensor (to inspect it, run <diag_obj>.model.pr_rho)
#> 
#>   Posteriors:
#>    - G_exp_lambda_f: [[0.02172369 1.08060381 1.08217007 1.08361545 1.08180345 1.08017684
#>   1.08183233 1.08032291 1.08267137 1.08084451 1.0794326  1.08208748
#>   1.08272802 1.08122456 1.12196327]]
#>    - G_exp_nu_f: 0.74
#>    - G_exp_theta_f: a (1, 50) tensor (to inspect it, run <diag_obj>.model.G_exp_theta_f)
#>    - rho_f: a (1, 50, 50, 15) tensor (to inspect it, run <diag_obj>.model.rho_f)
#> 
#> Optimisation:
#> 
#>    Elbo: 52373.250092620627
```

## Setup (Development mode)

Use this setup if you want to modify anything in the package. For
reproducibility reasons use the R version 4.1.2.

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
devtools::install()  # Install the current version of package
```
