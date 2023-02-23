# Contributing

When contributing to this repository, please first discuss the change you wish to make via issue, email, or any other method with the owners of this repository before making a change. 

## 📄 The Basics

<details><summary>📃 Building the documentation locally</summary>

### 📃 Building the documentation locally

If you just want to make minor edits to the text in the documentation, you don't need to go to the trouble of setting up your environment. You can just edit the files in the `docs/` folder and commit+push your changes.

You will just need to have [Quarto](https://quarto.org/) installed. To install Quarto, follow the instructions on the [Quarto website](https://quarto.org/docs/getting-started/installation.html). Ideally, use VSCode with the [Quarto extension](https://marketplace.visualstudio.com/items?itemName=quarto-dev.quarto-vscode) for a better experience.

After cloning the repository, cd to `docs/` and run `quarto preview . --render all --no-browser` to render the documentation locally. 

💡 If you want to change something in the structure of the documentation (say, the location of a menu or how the pages are laid out) take a look at the YAML file [docs/_quarto.yml](docs/_quarto.yml).

</details>

<details><summary>✋ How to contribute</summary>

### ✋ How to contribute

If you want to propose changes to the documentation if you were tasked to do something, you should follow the steps below:

1. Set up your environment by following the instructions in the [Dev Setup](#dev-setup) section.
2. Create a new branch from `develop` and give it a meaningful name. Best practices involve using the following format: `<your-username>/<issue-number>-<short-description>`. For example, if you are working on issue #3, you could name your branch `jonjoncardoso/3-fix-numpy-bug`.
    - Remember the [GitFlow](https://www.atlassian.com/git/tutorials/comparing-workflows/gitflow-workflow) workflow!
3. Make your changes and commit them to your branch. Remember to commit often and to write meaningful commit messages. If you are working on a specific issue, you can use the following format: `#<issue-number> <commit-message>`. For example, if you are working on issue #3, you could write `#3 Fix numpy bug`. 
4. When you are done, push all your commits and then open a [pull request](https://docs.github.com/en/github/collaborating-with-issues-and-pull-requests/creating-a-pull-request) to merge your branch into `develop`. You can do this by clicking on the "Compare & pull request" button on GitHub. Make sure to add a meaningful title and description to your pull request. If you are working on a specific issue, you can use the following format: `#<issue-number> <pull-request-title>`. For example, if you are working on issue #3, you could write `#3 Fix Numpy Bug`.

</details>


<details><summary>🔀 Our Git branches</summary>

### 🔀 Our Git branches

This repository has the following main branches:

- `main`: This is the "production" branch. It contains the latest version of the website. This branch is protected and only administrators can push to it. Whatever is in here, will be rendered and published on the website.
- `develop`: This is the "development" branch. It contains the latest version of the website under development. 
    - This branch is not protected and anyone with access to this repository can push to it. 
    - This is the branch you should be referring to when you want to make changes to the website. We might have updates to the website that are not yet ready to be published. In that case, we will push to `develop` and then an administrator will merge to `main` when we are ready to publish.
- `gh-pages`: This is the branch that contains the rendered version of the website. This branch is automatically updated by the GitHub workflow. **Do not push to this branch**.

However, you should not be working directly on `develop`. **Instead, you should create a new branch from `develop` and work on that branch**. When you are done, you should open a pull request to merge your branch into `develop`. This way, we can review your changes before merging them into `develop`. This is a good practice, often called [GitFlow](https://www.atlassian.com/git/tutorials/comparing-workflows/gitflow-workflow) that will help us avoid mistakes and also make it easier to revert changes if needed. 

💡 Not familiar with Git branches? Check out [this tutorial](https://www.atlassian.com/git/tutorials/using-branches), or [this one](https://www.youtube.com/watch?v=JTE2Fn_sCZs), and keep a bookmark for [this cheat sheet](https://www.atlassian.com/git/tutorials/atlassian-git-cheatsheet), or, perhaps the [Pro](https://git-scm.com/book/en/v2) Git book](https://git-scm.com/book/en/v2).

</details>

## 🧰 Dev Setup

Read on if you want a fuller setup to work on the documentation. This will allow you to run the documentation locally and also to make changes to the website structure.

<details><summary> 🔨 Use this if you just want to build the <strong>documentation</strong> locally </summary>

### 🔨 Use this if you just want to build the documentation locally

#### 📊 The R setup

1. Clone this repository to your computer.
2. Open a terminal and navigate to the root of this repository.
3. Although we recommend you have **R version 4.2.2** or higher, this should work with R>=3.6.0. You can check your R version by running `R --version` in your terminal.
4. Open the R console in this same directory and install `renv` package:
```r
install.packages("renv")
```
5. Run `renv::restore()` to install all the packages needed for this project. (No need to install packages manually.)
6. Run `renv::activate()` to activate the project environment. (No need to run this every time you open the project, just once is enough.)

#### The Quarto setup

1. Install [Quarto](https://quarto.org/docs/getting-started/installation.html) on your computer.
2. On the terminal, run the following command to start the website locally:
    ```bash
    quarto preview docs/ --render all --no-browser
    ```
    This will read the instructions from `_quarto.yml` and render the website locally.
5. Open your browser and navigate to `http://localhost:<port>/`. That's it!

</details>

<details><summary> 🔨 Use this if you want to make changes to the <strong>core Python package</strong> </summary>

### 🔨 Use this if you want to make changes to the core Python package
#### 🐍 The Python setup

1. Install [Python 3.8](python.org) or higher on your computer.
2. Install [anaconda](https://www.anaconda.com/products/individual) or [miniconda](https://docs.conda.io/en/latest/miniconda.html) on your computer.
3. Create a new conda environment:

    ```bash
    conda create -y -n=venv-vimure python=3.10.8
    ```
4. Activate the environment and make sure you have `pip` installed inside that environment:

    ```shell
    conda activate venv-vimure 
    ```

  Note: the `activate` command might vary depending on your OS.

💡 Remember to activate this particular `conda` environment whenever you reopen VSCode/the terminal.

10. Install required libraries

    ```shell
    pip install -r src/python/requirements.txt
    ```

Now, whenever you open a Jupyter Notebook, you should see the `venv-vimure` kernel available.

</details>

<details><summary> ⚒️ (Advanced) Jon's full setup </summary>

### ⚒️ (Advanced) Jon's full setup

⚠️ Proceed at your own risk ⚠️

I, [@jonjoncardoso](github.com/jonjoncardoso), like to use R on VSCode (WSL Ubuntu) instead of RStudio. It is a weird setup if you come from R, but it's a good setup for when you need to switch between R and Python all the time (the reality of this project). Feel free to just ignore this stuff but if you want to replicate my setup, just follow the steps below:

1. Install [VSCode](https://code.visualstudio.com/Download)
2. Install [WSL on Windows](https://learn.microsoft.com/en-us/windows/wsl/install) (or on your Mac)
3. Install the [WSL extension on VSCode](https://marketplace.visualstudio.com/items?itemName=ms-vscode-remote.remote-wsl) (if you are on Windows)
4. Open VSCode and open a new WSL window (Type `Ctrl+Shift+P` and type `WSL: New Window`). If on Mac, just open a new window.
6. Open the terminal on VSCode and install [R](https://cloud.r-project.org/)

**When doing R**

7. Install the [R extension on VSCode](https://marketplace.visualstudio.com/items?itemName=Ikuyadeu.r)
8. Install [Quarto](https://quarto.org/docs/getting-started/installation.html)
9. Install the [Quarto extension on VSCode](https://marketplace.visualstudio.com/items?itemName=quarto-dev.quarto-vscode)
10. When running R notebooks (either `.Rmd` or `.qmd`) manually, you will see that some plots do not render with the adequate size. To fix this, follow [these instructions](https://stackoverflow.com/a/70817205/843365).

**When doing Python**

11. Install the [Python extension on VSCode](https://marketplace.visualstudio.com/items?itemName=ms-python.python)
12. Install the [Jupyter extension on VSCode](https://marketplace.visualstudio.com/items?itemName=ms-toolsai.jupyter)

I also use the following VSCode Extensions:

- [GitHub Pull Requests and Issues](https://marketplace.visualstudio.com/items?itemName=GitHub.vscode-pull-request-github)
- [GitLens](https://marketplace.visualstudio.com/items?itemName=eamodio.gitlens)
- [GitHub Copilot](https://marketplace.visualstudio.com/items?itemName=GitHub.copilot)
- [Grammarly](https://marketplace.visualstudio.com/items?itemName=znck.grammarly)

</details>