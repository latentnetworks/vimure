# Python Dev Setup

**Author:** [@jonjoncardoso](https://jonjoncardoso.github.io)

Follow the instructions below if you are setting up your machine to further develop this Python project. If you are looking to just use the package (not work on its source), then head over to the 📦 [Installation](https://latentnetworks.github.io/vimure/latest/install.html) page.

## 📋 Requirements

- [Git](https://git-scm.com/downloads)
- [Python 3.8+](https://www.python.org/downloads/)
- [miniconda](https://docs.conda.io/en/latest/miniconda.html) (preferred) or [anaconda](https://www.anaconda.com/products/individual)

## 🚀 Setup

1. Clone the repository, change to the `develop` branch, then navigate to the root directory of the Python package:

    ```bash
    git clone git@github.com:latentnetworks/vimure.git
    git checkout develop
    cd vimure/src/python
    ```

(I recommend you open `vimure/src/python` on VSCode to have the same environment as me)

2. Create a separate git branch for your changes:

    ```bash
    git checkout -b <your-branch-name>
    git push -u origin <your-branch-name>
    ```

2. Create a conda environment and install the dependencies:

    ```bash
    conda create -n venv-vimure
    conda activate venv-vimure
    pip install -r requirements_dev.txt
    ```

3. Install the package in editable mode to test it locally:

    ```bash
    pip install -e .
    ```

4. You can also run the tests:

    ```bash
    pytest
    ```

5. Now, make your changes by editing the .py files directly. Test your changes (bug fixes, improvements, new features) on the terminal or in a Jupyter Notebook.  If you add new features (new functions, new parameters), please also add new unit tests to test them.

6. Once you're happy with your changes, open a Pull Request and tag @jonjoncardoso as a reviewer.