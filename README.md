# VIMuRe

![Github Badge](https://github.com/latentnetworks/vimure/workflows/test/badge.svg)
[![codecov](https://codecov.io/gh/latentnetworks/vimure/branch/main/graph/badge.svg?token=NAZP90D12J)](https://codecov.io/gh/latentnetworks/vimure)

Latent Network Models to Account for Noisy, Multiply-Reported Social Network Data.


If you use this code please cite this article (preprint).

> De Bacco C, Contisciani M, Cardoso-Silva J, Safdari H, Baptista D, Sweet T, Young JG, Koster J, Ross CT, McElreath R, Redhead D. Latent Network Models to Account for Noisy, Multiply-Reported Social Network Data. arXiv preprint [arXiv:2112.11396](https://arxiv.org/abs/2112.11396). 2021.


# Notebooks

To understand how you could use the code or to replicate our research, check the notebooks folder:

## Python

- [`Notebook 01 - Generate Synthetic Networks`](https://github.com/latentnetworks/vimure/blob/main/notebooks/Notebook%2001%20-%20Generate%20Synthetic%20Networks.ipynb)
- [`Notebook 02 - Run Model`](https://github.com/latentnetworks/vimure/blob/main/notebooks/Notebook%2002%20-%20Run%20Model.ipynb)
- [`Notebook 03 - Experiment Under and Over Reporters`](https://github.com/latentnetworks/vimure/blob/main/notebooks/Notebook%2003%20-%20Experiment%20under%20and%20over%20reporters.ipynb)
- [`Notebook 04 - Karnataka data - Single Run`](https://github.com/latentnetworks/vimure/blob/main/notebooks/Notebook%2004%20-%20Karnataka%20data%20-%20Single%20Run.ipynb)
- [`Notebook 04 - Karnataka data (Full)`](https://github.com/latentnetworks/vimure/blob/main/notebooks/Notebook%2004%20-%20Karnataka%20data.ipynb)
- [`Notebook 05 - Experiment varying reciprocity`](https://github.com/latentnetworks/vimure/blob/main/notebooks/Notebook%2005%20-%20Experiment%20varying%20reciprocity.ipynb)
- [`Notebook 06 - Nicaragua data`](https://github.com/latentnetworks/vimure/blob/main/notebooks/Notebook%2006%20-%20Nicaragua%20data.ipynb)

## R

- [`RMarkdown 01 - Generate Synthetic Networks`](https://github.com/latentnetworks/vimure/blob/develop/notebooks/R/RMarkdown%201%20-%20%20Generate%20Synthetic%20Networks.Rmd)
- [`RMardown 02 - Run Model`](https://github.com/latentnetworks/vimure/blob/develop/notebooks/R/RMarkdown%202%20-%20Run%20Model.Rmd)
- [`RMardown 03 - Read, Parse and Fit Vimure on Karnataka data`](https://github.com/latentnetworks/vimure/blob/develop/notebooks/R/RMarkdown%203%20-%20Karnataka%20Data.Rmd)

# Setup

Instructions on how to replicate our research either downloading our package via pip or by using the Virtualenv setup used for developing the package.

## Alternative 01: using pip

Install `vimure==0.1` with the following command:

```console
pip install "git+https://github.com/latentnetworks/vimure.git#egg=vimure&subdirectory=src/python/"
```

## Alternative 02: using Virtualenv

If you are familiar with Virtualenv, you could use the Virtualenv provided in this repository. Use this setup if you want to modify anything in the package.

1. [Clone the repository](https://docs.github.com/en/github/creating-cloning-and-archiving-repositories/cloning-a-repository-from-github/cloning-a-repository) to a directory in your machine
2. [Set up Python, Pip and Virtualenv](http://timsherratt.org/digital-heritage-handbook/docs/python-pip-virtualenv/)
3. Open the terminal and build the project:
```{console}
cd vimure
make venv-build  # Create a virtualenv
```
The first time you run this build, it will take several minutes to complete. Trust me, it is better to run this and wait the building time than having to install each multiple python dependencies by hand and having to figure out why your colleague gets a weird and mysterious, previously unseen Exception when running the same code as you!

4. Still on the terminal, run Jupyter server with the command:
```{console}
make venv-up
```

A URL will show up on your screen, either click on it or copy-paste to your browser and run the notebooks.

Or you can run step by step as below:
1. Run the following command in `vimure/` directory:
```{bash}
virtualenv venv
```
2. Activate the virtualenv by running the following command (you always need to run this command before running any other command in the virtualenv):
```{bash}
source venv/bin/activate
```
3. Install the required packages by running the following command:
```{bash}
pip install -r src/python/requirements.txt
pip install -e src/python/.   # Install Vimure package
```
4. Create a JupyterLab instance by running the following command:
```{bash}
jupyter lab
```

# License

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NON INFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
