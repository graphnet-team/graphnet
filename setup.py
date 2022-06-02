# inspiration: https://github.com/PyTorchLightning/pytorch-lightning
try:
    from setuptools import setup, find_packages
except ImportError:
    from distutils.core import setup

import subprocess
import platform
import sys
import versioneer

# Utility method(s)
def install(package):
    subprocess.check_call(
        [sys.executable, "-m", "pip", "install", "--user", package]
    )

# from https://github.com/PyTorchLightning/pytorch-lightning
def _load_requirements(path_dir: str , file_name: str = 'requirements.txt', comment_char: str = '#') -> List[str]:
    """Load requirements from a file
    >>> _load_requirements(PROJECT_ROOT)  # doctest: +ELLIPSIS +NORMALIZE_WHITESPACE
    ['numpy...', 'torch...', ...]
    """
    with open(os.path.join(path_dir, file_name), 'r') as file:
        lines = [ln.strip() for ln in file.readlines()]
    reqs = []
    for ln in lines:
        # filer all comments
        if comment_char in ln:
            ln = ln[:ln.index(comment_char)].strip()
        # skip directly installed dependencies
        if ln.startswith('http'):
            continue
        if ln:  # if requirement is not empty
            reqs.append(ln)
    return reqs

#def install_requirement(requirement):
#    subprocess.check_call(
#        [sys.executable, "-m", "pip", "install", "--user", "-r", requirement]
#    )

with open('requirements.txt') as dependencies_file:
    dependencies = [x.rstrip("\n") for x in dependencies_file.readlines()]

# Requirements definitions
SETUP_REQUIRES = [
    "setuptools == 59.5.0",
]

INSTALL_REQUIRES = [
    "sqlalchemy",
    "pandas>=1.1.0",
    "numpy",
    "timer",
    "tqdm",
    "torch-cluster==1.5.9",
    "torch-scatter==2.0.9",
    "torch-sparse==0.6.12",
    "torch-spline-conv==1.2.1",
    "torch-geometric==2.0.1",
    "pytorch-lightning==1.5.6",
    "dill",
    "wandb",
    "matplotlib",
]

with open('require.txt') as dependencies_file:
    depend = [x.rstrip("\n") for x in dependencies_file.readlines()]
INSTALL_REQUIRES = [depend]

EXTRAS_REQUIRE = {
    "develop": [
        "black",
        "pytest",
        "pylint",
        "pydocstyle",
        "coverage",
        "anybadge",
        "pre-commit",
        "sphinx",
        "sphinx_rtd_theme",
        "versioneer",
    ],
    "build": [
        "pytest",
    ]
}

# https://pypi.org/classifiers/
CLASSIFIER = {
    "Development Status :: 3 - Alpha",
    "Programming Language :: Python :: 3.8",
    "Programming Language :: Python :: 3.9",
    "Environment :: CPU",
    "Environment :: GPU",
}
# Ensure pytorch is already installed (see e.g.
# https://github.com/pyg-team/pytorch_geometric/issues/861#issuecomment-566424944)
#try:
#    import torch  # pyright: reportMissingImports=false
#except ImportError:
#    install("torch==1.10.1")

# Installs individual packages, does not '--find_links' into account
#for req in dependencies:
#    if not req.startswith("#"):
#        try:
#            import req
#        except ImportError:
#            install(str(req))

# directly install requirements.txt file
#try:
#    # try to import the needed packages
#    for req in dependencies:
#        if not req.startswith("#"):
#            import req
# if any package is missing install the requirements.txt file
#except ImportError:
#    install_requirement("requirements.txt")

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
    #install_requires=INSTALL_REQUIRES, # requires a list without --find_links
    install_requires=_load_requirements(PATH_ROOT),
    extras_require=EXTRAS_REQUIRE,
    #classifier=CLASSIFIER,
    dependency_links=[
        "https://download.pytorch.org/whl/torch_stable.html#egg=torch==1.10.1+cu113"
        "https://data.pyg.org/whl/torch-1.10.0+cu113.html#egg=torch_scatter~=2.0.9",
    ], # maybe needs an additonal flag 'pip install -e .[develop] --allow-all-external'
)

