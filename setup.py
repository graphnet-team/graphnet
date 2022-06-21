import subprocess
import sys
from setuptools import setup, find_packages
import versioneer

# Requirements definitions
SETUP_REQUIRES = [
    "setuptools>=59.5.0",
]

INSTALL_REQUIRES = [
    "sqlalchemy>=1.4",
    "pandas>=1.3",
    "numpy>=1.21",
    "timer>=0.2",
    "tqdm>=4.64",
    "dill>=0.3",
    "wandb>=0.12",
    "matplotlib>=3.5",
    "scikit_learn>=1.0",
    "scipy>=1.8",
    "torch==1.11",
    "torch-cluster==1.6.0",
    "torch-scatter==2.0.9",
    "torch-sparse==0.6.13",
    "torch-spline-conv==1.2.1",
    "torch-geometric==2.0.4",
    "pytorch-lightning>=1.6",
]

EXTRAS_REQUIRE = {
    "develop": [
        "black",
        "colorlog",
        "coverage",
        "pre-commit",
        "pydocstyle",
        "pylint",
        "pytest",
        "sphinx",
        "sphinx_rtd_theme",
        "versioneer",
    ]
}

# https://pypi.org/classifiers/
CLASSIFIERS = [
    "Development Status :: 3 - Alpha",
    "Programming Language :: Python :: 3.6",
    "Programming Language :: Python :: 3.7",
    "Programming Language :: Python :: 3.8",
    "Programming Language :: Python :: 3.9",
    "Programming Language :: Python :: 3.10",
    "Environment :: CPU",
    "Environment :: GPU",
]

setup(
    name="graphnet",
    version=versioneer.get_version(),
    description=(
        "A common library for using graph neural networks (GNNs) in netrino "
        "telescope experiments."
    ),
    license="Apache 2.0",
    author="The IceCube Collaboration",
    url="https://github.com/icecube/graphnet",
    cmdclass=versioneer.get_cmdclass(),
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    setup_requires=SETUP_REQUIRES,
    install_requires=INSTALL_REQUIRES,
    extras_require=EXTRAS_REQUIRE,
    classifiers=CLASSIFIERS,
)
