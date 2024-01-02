from setuptools import setup, find_packages

setup(
    name="PPO-PyTorch-Library",
    version="0.1.0",
    description="A modularized version nikhilbarhate99/PPO-PyTorch",
    author="Mahyar Riazati",
    author_email="mr.riazati1999@gmail.com",
    url="https://github.com/mhyrzt/PPO-PyTorch",
    packages=find_packages(),
    install_requires=[
        "torch",
        "numpy",
        "pandas",
        "Pillow",
        "seaborn",
        "gymnasium",
        "matplotlib",
    ],
)
