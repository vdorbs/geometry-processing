from setuptools import find_packages, setup

setup(
    name='geometry_processing',
    packages=find_packages(),
    install_requires=[
        'matplotlib',
        'potpourri3d',
        'torch',
        'torchsparsegradutils'
    ]
)
