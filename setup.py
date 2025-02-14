from setuptools import setup, find_packages
from cleansweep.__version__ import __version__

setup(
    name='cleansweep',
    version=__version__,
    description='CleanSweep - variant calling with plate swipe data.',
    author='Marco Teixeira',
    author_email='mcarvalh@broadinstitute.org',
    license='MIT License',
    packages=find_packages(
        where='cleansweep'
    ),
    package_dir={'': 'cleansweep'},
    include_package_data=True,
    install_requires=[],
    classifiers=[
        'MIT License',  
        'Programming Language :: Python :: 3.12',
    ]
)