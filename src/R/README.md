# vimure

![Github Badge](https://github.com/latentnetworks/vimure/workflows/test/badge.svg)
[![codecov](https://codecov.io/gh/latentnetworks/vimure/branch/main/graph/badge.svg?token=NAZP90D12J)](https://codecov.io/gh/latentnetworks/vimure)

Latent Network Models to Account for Noisy, Multiply-Reported Social Network Data.

If you use this code, please cite this article:

> _Caterina De Bacco, Martina Contisciani, Jonathan Cardoso-Silva, Hadiseh Safdari, Gabriela Lima Borges, Diego Baptista, Tracy Sweet, Jean-Gabriel Young, Jeremy Koster, Cody T Ross, Richard McElreath, Daniel Redhead, Eleanor A Power, **Latent network models to account for noisy, multiply reported social network data**, Journal of the Royal Statistical Society Series A: Statistics in Society, 2023;, qnac004, [https://doi.org/10.1093/jrsssa/qnac004](https://doi.org/10.1093/jrsssa/qnac004)_

VIMuRe package is available in R and Python. You are looking at the R documentation.

## Installation

```r
install.packages("devtools")
devtools::install_github("latentnetworks/vimure", subdir="src/R", ref="main")
```

## Feedback

- by taking this [survey](https://forms.gle/QaK5AWWYy78jZfyR6) or,
- by [opening an issue](https://github.com/latentnetworks/vimure/issues/new/choose) on Github.

Note that if reticulate did not find a non-system installation of python you may be prompted if you want it to download and install miniconda. Miniconda is the recommended installation method for most users, as it ensures that the R python installation is isolated from other python installations. If you initially declined the miniconda installation prompt, you can later manually install miniconda by running `reticulate::install_miniconda()`

``` r
library(vimure)
vimure::install_vimure()
```

``` text
    #> Using virtual environment '/home/gabriela-borges/Git/vimure/Python/venv' ...
    #> 
    #> Installation complete.
```

You can confirm that the installation succeeded with:

``` r
library(vimure)
vimureP$utils$is_sparse(matrix(c(1:50), ncol=10))
```

```text
    #> [1] FALSE
```

This will provide you with a default installation of VIMuRe suitable for use with the vimure R package.

## Alternate Versions (advanced mode)

VIMuRe is distributed as a Python package and so needs to be installed within a Python environment on your system. By default, the
install_vimure() function attempts to install VIMuRe within an isolated Python environment (“r-reticulate”).

Note that `install_vimure()` isn’t required to use VIMuRe with the package. If you manually configure a python environment with the required dependencies, you can tell R to use it by pointing reticulate at it, commonly by setting an environment variable:

``` r
  Sys.setenv("RETICULATE_PYTHON" = "~/path/to/python-env/bin/python")
```

By default, `install_vimure()` install the latest *main* branch of VIMuRe You can override this behavior by specifying the version parameter. For example:

``` r
install_vimure(version = "master")
```

You can also install a local version of VIMuRe by specifying a URL/Path to a VIMuRe binary. For example:

``` r
install_vimure(version = "~/Git/vimure/src/python")
```

# Quick start

## Generating Ground Truth - `Y` and Observed Network - `X`

Simply create an object with the desired synthetic network class:

``` r
random_net <- gm_crep(N=100, M=100, L=1, C=2, avg_degree=10, sparsify=T, eta=0.99, seed=10)
Y <- extract_Y(random_net)
X <- build_X(random_net, flag_self_reporter=T, cutoff_X=F, seed=10L)

paste("Reciprocity Y:", round(overall_reciprocity(Y[1,,]),3))
```

```text

    #> [1] "Reciprocity Y: 0.788"

```

``` r
paste("Reciprocity X (intersection):", round(overall_reciprocity(random_net$X_intersection$toarray()[1,,]),3))
```

```text

    #> [1] "Reciprocity X (intersection): 0.688"

```

``` r
paste("Reciprocity X (union):", round(overall_reciprocity(random_net$X_union$toarray()[1,,]),3))
```

```text

    #> [1] "Reciprocity X (union): 0.722"

```

## Python interface

`vimure` is a R binding of a Python package. Many Python basic objects are quickly converted to R automatically. Custom Python objects that can
not be converted automatically are stored in R as a `python.builtin.object`. As a `python.builtin.object`, you can access all object’s attributes as it is in Python using the dollar sign `$`.

Use the function `class` to check if a object is stored in Python.

``` r
class(random_net)
```

```text

    #> [1] "vimure.synthetic.Multitensor"         
    #> [2] "vimure.synthetic.StandardSBM"         
    #> [3] "vimure.synthetic.BaseSyntheticNetwork"
    #> [4] "vimure.io.BaseNetwork"                
    #> [5] "python.builtin.object"

```

`random_net` is stored as a Python object. You can access its attributes using the dollar sign `$` or using our `extract_*` functions which always will return a R object.

## Run Model

In R we can construct and fit the vimure model by using the `vimure` function. The `vimure` function inherit all arguments from the original `VimudeModel` class and `VimureModel.fit()`. See more info about arguments in `help(vimure)`.

``` r
model <- vimure(random_net$X, random_net$R, mutuality=T, K=2, num_realisations=1, max_iter=150)
diag <- summary(model)
```

```text

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
    #>    - rho:    a (1, 100, 100, 2) tensor (to inspect it, run <diag_obj>.model.pr_rho)
    #> 
    #>   Posteriors:
    #>    - G_exp_lambda_f: [[0.00151633 1.43093229]]
    #>    - G_exp_nu_f: 0.77
    #>    - G_exp_theta_f: a (1, 100) tensor (to inspect it, run <diag_obj>.model.G_exp_theta_f)
    #>    - rho_f: a (1, 100, 100, 2) tensor (to inspect it, run <diag_obj>.model.rho_f)
    #> 
    #> Optimisation:
    #> 
    #>    Elbo: 1120.418888338589
```
