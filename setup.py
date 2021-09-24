from setuptools import setup, find_packages

setup(
    name='gnn_reco',
    version='0.1.1',   
    description='A common library for using graph neural networks (GNNs) in netrino telescopes.',
    url='https://github.com/icecube/gnn-reco',
    author='The IceCube Collaboration',
    license='MIT',
    packages=['gnn_reco'],
    package_dir={'': 'src'},
    install_requires=[
        'sqlalchemy',
        'pandas',
        'numpy',
        'tqdm',
        'pytest',
        'pylint',
        'coverage',
        'anybadge',
        ],
)