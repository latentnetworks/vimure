# Using a default Docker image obtained from https://jupyter-docker-stacks.readthedocs.io/en/latest/index.html
FROM jupyter/scipy-notebook:notebook-6.4.0

USER root

RUN apt update  && apt install -y vim

USER jovyan

# This is what will be the default directory inside the Docker container #
WORKDIR /mnt/code

COPY requirements.txt requirements.txt
RUN pip install --upgrade pip
RUN pip install -r requirements.txt

RUN jupyter labextension install @jupyterlab/toc-extension
