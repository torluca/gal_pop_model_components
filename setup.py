#!/usr/bin/env python

import os
import sys

try:
    from setuptools import setup, find_packages
except ImportError:
    from distutils.core import setup


if sys.argv[-1] == 'publish':
    os.system('python setup.py sdist upload')
    sys.exit()

readme = open('README.rst').read()
doclink = """
Documentation
-------------

The full documentation is at http://gal_pop_model_components.rtfd.org."""
history = open('HISTORY.rst').read().replace('.. :changelog:', '')

setup(
    name='gal_pop_model_components',
    version='1.0.0',
    description='Package containing libraries of galaxy population modelling components',
    long_description=readme + '\n\n' + doclink + '\n\n' + history,
    author='Luca Tortorelli',
    author_email='Luca.Tortorelli@physik.lmu.de',
    url='https://gitlab.physik.uni-muenchen.de/acai/gal_pop_model_components',
    packages=[
        'gal_pop_model_components',
    ],
    package_dir={'gal_pop_model_components': 'gal_pop_model_components'},
    include_package_data=True,
    install_requires=['numpy', 'scipy', 'astropy'
    ],
    license='MIT',
    zip_safe=False,
    keywords='gal_pop_model_components',
    classifiers=[
        'Development Status :: 2 - Pre-Alpha',
        'Intended Audience :: Developers',
        'License :: OSI Approved :: MIT License',
        'Natural Language :: English',
        'Programming Language :: Python :: 2',
        'Programming Language :: Python :: 2.6',
        'Programming Language :: Python :: 2.7',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.3',
        'Programming Language :: Python :: Implementation :: PyPy',
    ],
)
