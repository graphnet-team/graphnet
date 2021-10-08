import subprocess
import sys
from setuptools import setup, find_packages

# Utility method(s)
def install(package):
    subprocess.check_call([sys.executable, "-m", "pip", "install", package])

# Requirements definitions
EXTRAS_REQUIRE = {
    'develop': [
        'pytest',
        'pylint',
        'coverage',
        'anybadge',
        'sphinx',
        'sphinx_rtd_theme',
    ],
}

INSTALL_REQUIRES = [
    'sqlalchemy',
    'pandas',
    'numpy',
    'tqdm',
    'torch-scatter',
    'torch-sparse',
    'torch-cluster',
    'torch-spline-conv',
    'torch-geometric',
]

# Ensure pytorch is already installed (see e.g. https://github.com/pyg-team/pytorch_geometric/issues/861#issuecomment-566424944)
try:
    import torch  # pyright: reportMissingImports=false
except ImportError:
    install('torch>=1.9.0')

setup(
    name='gnn_reco',
    version='0.1.1',   
    description='A common library for using graph neural networks (GNNs) in netrino telescopes.',
    url='https://github.com/icecube/gnn-reco',
    author='The IceCube Collaboration',
    license='MIT',
    packages=find_packages(where='src'),
    package_dir={'': 'src'},
    install_requires=INSTALL_REQUIRES,
    extras_require=EXTRAS_REQUIRE,
    dependency_links=[
        'https://data.pyg.org/whl/torch-1.9.0+cpu.html',
    ],
)