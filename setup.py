import subprocess
import sys
from setuptools import setup, find_packages
import versioneer

# Utility method(s)
def install(package):
    subprocess.check_call([sys.executable, "-m", "pip", "install", "--user", package])

# Requirements definitions
EXTRAS_REQUIRE = {
    'develop': [
        'pytest',
        'pylint',
        'coverage',
        'anybadge',
        'sphinx',
        'sphinx_rtd_theme',
        'versioneer',
    ],
}

INSTALL_REQUIRES = [
    'sqlalchemy',
    'pandas',
    'numpy',
    'timer',
    'tqdm',
    'torch-cluster==1.5.9',
    'torch-scatter==2.0.9',
    'torch-sparse==0.6.12',
    'torch-spline-conv==1.2.1',
    'torch-geometric==2.0.1',
    'pytorch-lightning',
    'dill',
    'wandb',
    'matplotlib',
]

# Ensure pytorch is already installed (see e.g. https://github.com/pyg-team/pytorch_geometric/issues/861#issuecomment-566424944)
try:
    import torch  # pyright: reportMissingImports=false
except ImportError:
    install('torch==1.9.0')

setup(
    name='graphnet',
    version=versioneer.get_version(),
    description='A common library for using graph neural networks (GNNs) in netrino telescope experiments.',
    license='Apache 2.0',
    author='The IceCube Collaboration',
    url='https://github.com/icecube/graphnet',
    cmdclass=versioneer.get_cmdclass(),
    packages=find_packages(where='src'),
    package_dir={'': 'src'},
    install_requires=INSTALL_REQUIRES,
    extras_require=EXTRAS_REQUIRE,
    dependency_links=[
        'https://data.pyg.org/whl/torch-1.9.0+cpu.html',
    ],
)
