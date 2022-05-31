
# Python Setup

Instructions on how to replicate our research either downloading our package by using the Virtualenv setup used for developing the package.

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