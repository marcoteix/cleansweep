from setuptools import setup, find_packages
from matiss.__init__ import __version__

setup(
    name='cleansweep',
    version=__version__,
    description='CleanSweep - variant calling with plate swipe data.',
    author='Marco Teixeira',
    author_email='mcarvalh@broadinstitute.org',
    license='MIT License',
    packages=find_packages(where='cleansweep'),
    package_dir={'': 'cleansweep'},
    install_requires=[],
    classifiers=[
        'Development Status :: 3 - Alpha',
        'Intended Audience :: Science/Research',
        'MIT License',  
        'Programming Language :: Python :: 3.8',
    ],
    scripts = [
        "cleansweep/downsample_vcf.sh"
    ]
)