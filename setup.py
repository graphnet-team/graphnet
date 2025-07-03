# mypy: disable-error-code="no-untyped-call"
"""Setup script for the GraphNeT package."""

from setuptools import setup, find_packages
import versioneer

# Requirements definitions
SETUP_REQUIRES = [
    "setuptools>=68.2.2",
]

INSTALL_REQUIRES = [
    "colorlog>=6.6",
    "configupdater",
    "dill>=0.3",
    "matplotlib>=3.5",
    "numpy>=1.22,<2.0",
    "pandas>=1.3",
    "pyarrow",
    "pydantic",
    "ruamel.yaml",
    "scikit_learn>=1.0",
    "scipy>=1.7",
    "sqlalchemy>=1.4",
    "tqdm>=4.64",
    "wandb>=0.12",
    "polars >=0.19",
    "torchscale==0.2.0",
    "h5py>= 3.7.0",
]

EXTRAS_REQUIRE = {
    "develop": [
        "black",
        "coverage",
        "docformatter",
        "MarkupSafe<=2.1",
        "mypy",
        "myst-parser",
        "pre-commit<4.0",
        "pydocstyle",
        "pylint",
        "pytest",
        "pytest-order",
        "sphinx",
        "sphinx-material",
        "sphinx-autodoc-typehints",
        "versioneer",
        "flake8",
    ],
    # --- PyTorch 2.5.0 ---
    "torch-25": [
        "torch==2.5",
        "torch-geometric",
        "pyg_lib",
        "torch_scatter",
        "torch_sparse",
        "torch_cluster",
        "torch_spline_conv",
        "pytorch-lightning>=2.0",
    ],
    # --- PyTorch 2.6.0 ---
    "torch-26": [
        "torch==2.6",
        "torch-geometric",
        "pyg_lib",
        "torch_scatter",
        "torch_sparse",
        "torch_cluster",
        "torch_spline_conv",
        "pytorch-lightning>=2.0",
    ],
    # --- PyTorch 2.7.0 ---
    "torch-27": [
        "torch==2.7",
        "torch-geometric",
        "pyg_lib",
        "torch_scatter",
        "torch_sparse",
        "torch_cluster",
        "torch_spline_conv",
        "pytorch-lightning>=2.0",
    ],
}

# https://pypi.org/classifiers/
CLASSIFIERS = [
    "Development Status :: 3 - Alpha",
    "Intended Audience :: Developers",
    "Intended Audience :: Science/Research",
    "Programming Language :: Python :: 3.8",
    "Programming Language :: Python :: 3.9",
    "Programming Language :: Python :: 3.10",
    "Programming Language :: Python :: 3.11",
    "Environment :: CPU",
    "Environment :: GPU",
    "License :: OSI Approved :: Apache Software License",
    "Topic :: Scientific/Engineering",
    "Topic :: Scientific/Engineering :: Physics",
    "Topic :: Scientific/Engineering :: Mathematics",
    "Topic :: Scientific/Engineering :: Artificial Intelligence",
    "Topic :: Software Development",
    "Topic :: Software Development :: Libraries",
    "Topic :: Software Development :: Libraries :: Python Modules",
]

setup(
    name="graphnet",
    version=versioneer.get_version(),
    description=(
        "A common library for using deep learning in neutrino "
        "telescope experiments."
    ),
    license="Apache 2.0",
    author="The GraphNeT development team",
    url="https://github.com/graphnet-team/graphnet",
    cmdclass=versioneer.get_cmdclass(),
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    setup_requires=SETUP_REQUIRES,
    install_requires=INSTALL_REQUIRES,
    extras_require=EXTRAS_REQUIRE,
    classifiers=CLASSIFIERS,
)
