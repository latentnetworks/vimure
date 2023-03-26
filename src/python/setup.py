from setuptools import setup, find_packages

setup(
    name="vimure",
    description="VIMuRe - A Latent Network Model for Multiply-Reported Social Network Data",
    long_description=(
        "VIMuRe - A Single-Layered Latent Network Models to Account for Noisy, Multiply-Reported Social Network Data"
    ),
    version="0.1.0-dev-abc8820",
    py_modules=["vimure"],
    author="Caterina de Baco",
    keywords=[""],
    install_requires=[
        "numpy <= 1.24.2",
        "scipy <= 1.10.1",
        "pandas < 2.0.0",
        "scikit-learn <= 1.2.1",
        "scikit-tensor-py3 @ https://github.com/jonjoncardoso/scikit-tensor-py3/archive/master.zip",
        "networkx < 3.0.0"
    ],
    packages=find_packages(),
    python_requires=">=3",
    classifiers=[
        "Programming Language :: Python :: 3"
    ],
)
