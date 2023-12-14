from setuptools import setup, find_packages

setup(
    name='fplibrary',
    version='0.1',
    packages=find_packages(),
    author='Viktoriia Yuzkiv, Angelo Di Gianvito, Oliver Gatland, Joaquin Ossa',
    install_requires=[
        'numpy>=1.20.3',
        'pandas>=1.2.4',
        'scipy>=1.7.3',
        'matplotlib>=3.5.1',
        'seaborn',
        'scikit-learn==1.3.1',
        'setuptools==68.0.0',
        'openmeteo-requests',
        'requests-cache retry-requests'
    ],
)