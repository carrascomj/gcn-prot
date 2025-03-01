from setuptools import find_packages, setup

setup(
    name="gcn_prot",
    packages=find_packages(),
    install_requires=[
        "matplotlib",
        "mpi4py",
        "numpy",
        "pandas",
        "sklearn",
        "scipy",
        "torch",
        "wget",
    ],
    version="0.3.0",
    description="Graph convolutional networks to perform structural learning of proteins",
    author="Jorge Carrasco and Bjorn Hansen",
    license="MIT",
)
