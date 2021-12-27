from setuptools import setup, find_packages

setup(
    name="vimure",
    description="VIMuRe - A Latent Network Model for Multiply-Reported Social Network Data",
    long_description=(
        "VIMuRe - A Single-Layered Latent Network Models to Account for Noisy, Multiply-Reported Social Network Data"
    ),
    version="0.1",
    py_modules=["vimure"],
    author="Caterina de Baco",
    keywords=[""],
    install_requires=[
        "numpy >= 1.13",
        "scipy >= 0.19.1",
        "pandas >= 0.20.3",
        "scikit-learn >= 0.19.0",
        "scikit-tensor-py3 @ https://github.com/evertrol/scikit-tensor-py3/archive/99190292b9564e7c4a9312280db732ceedf7b7e3.zip",
        "networkx >= 2.0.0",
        "plotnine >= 0.7.0"
    ],
    packages=find_packages(),
    python_requires=">=3",
    classifiers=[
        "Programming Language :: Python :: 3.5",
        "Programming Language :: Python :: 3.6",
    ],
)
