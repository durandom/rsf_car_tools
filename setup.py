from setuptools import setup, find_packages

setup(
    name="powersteering",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "numpy",
        "scikit-learn", 
        "plotext",
        "rich",
        "loguru",
        "configobj"
    ],
    entry_points={
        'console_scripts': [
            'powersteering=powersteering.cli:main',
        ],
    },
)
