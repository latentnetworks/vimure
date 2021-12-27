# VIMuRe

![Github Badge](https://github.com/latentnetworks/vimure/workflows/test/badge.svg)
[![codecov](https://codecov.io/gh/latentnetworks/vimure/branch/main/graph/badge.svg?token=NAZP90D12J)](https://codecov.io/gh/latentnetworks/vimure)

Latent Network Models to Account for Noisy, Multiply-Reported Social Network Data.

# Notebooks

To understand how you could use the code or to replicate our research, check the notebooks folder:

- [`Notebook 01 - Generate Synthetic Networks`](https://github.com/latentnetworks/vimure/blob/main/notebooks/Notebook%2001%20-%20Generate%20Synthetic%20Networks.ipynb)
- [`Notebook 02 - Run Model`](https://github.com/latentnetworks/vimure/blob/main/notebooks/Notebook%2002%20-%20Run%20Model.ipynb)
- [`Notebook 03 - Experiment Under and Over Reporters`](https://github.com/latentnetworks/vimure/blob/main/notebooks/Notebook%2003%20-%20Experiment%20under%20and%20over%20reporters.ipynb)
- [`Notebook 04 - Karnataka data - Single Run`](https://github.com/latentnetworks/vimure/blob/main/notebooks/Notebook%2004%20-%20Karnataka%20data%20-%20Single%20Run.ipynb)
- [`Notebook 04 - Karnataka data (Full)`](https://github.com/latentnetworks/vimure/blob/main/notebooks/Notebook%2004%20-%20Karnataka%20data.ipynb)
- [`Notebook 05 - Experiment varying reciprocity`](https://github.com/latentnetworks/vimure/blob/main/notebooks/Notebook%2005%20-%20Experiment%20varying%20reciprocity.ipynb)
- [`Notebook 06 - Nicaragua data`](https://github.com/latentnetworks/vimure/blob/main/notebooks/Notebook%2006%20-%20Nicaragua%20data.ipynb)

# Setup

Instructions on how to replicate our research either downloading our package via pip or by using the Docker setup used for developing the package.

## Alternative 01: using pip

Install `vimure==0.1` with the following command:

```console
pip install git+https://github.com/latentnetworks/vimure.git#egg=vimure\&subdirectory=src/python/
```

## Alternative 02: using Docker

If you are familiar with Docker, you could use the Docker containers provided in this repository. Use this setup if you want to modify anything in the package.

1. [Clone the repository](https://docs.github.com/en/github/creating-cloning-and-archiving-repositories/cloning-a-repository-from-github/cloning-a-repository) to a directory in your machine
2. Install [Docker](https://docs.docker.com/get-docker/) and [docker-compose](https://docs.docker.com/compose/install/) on your machine
3. Open the terminal and build the project:
```{console}
cd vimure
docker-compose build
```
The first time you run this build, it will take several minutes to complete. Trust me, it is better to run this and wait the building time than having to install each multiple python dependencies by hand and having to figure out why your colleague gets a weird and mysterious, previously unseen Exception when running the same code as you!

4. Still on the terminal, run Jupyter server with the command:
```{console}
docker-compose up
```

A URL will show up on your screen, either click on it or copy-paste to your browser and run the notebooks.

# License

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NON INFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
