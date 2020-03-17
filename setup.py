from setuptools import find_packages, setup

setup(
    name="gcn_prot",
    packages=find_packages(),
    install_requires=[
        "numpy",
        "mpi4py",
        "pandas",
        "sklearn",
        "scipy",
        "torch",
        "torchvision",
        "wget",
    ],
    version="0.1.0",
    description="Graph convolutional networks tperform structural learning of proteins",
    author="Jorge Carrasco and Bjorn Hansen",
    license="MIT",
)
