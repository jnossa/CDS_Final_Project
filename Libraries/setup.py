from setuptools import setup, find_packages

setup(
    name='fplibrary',
    version='0.1',
    packages=find_packages(),
    install_requires=[
        'numpy>=1.20.3',
        'pandas>=1.2.4',
        'scipy>=1.7.3',
        'matplotlib>=3.5.1'
    ],
)