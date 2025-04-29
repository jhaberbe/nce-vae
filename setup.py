# -*- coding: utf-8 -*-

# Learn more: https://github.com/kennethreitz/setup.py

from setuptools import setup, find_packages


with open('README.md') as f:
    readme = f.read()

with open('LICENSE') as f:
    license = f.read()

setup(
    name='ncvae',
    version='0.1.0',
    description='Noise Contrastive Estimation',
    long_description=readme,
    author='James Haberberger',
    author_email='jhaberbe@stanford.edu',
    url='https://github.com/jhaberbe/ncvae',
    license=license,
    packages=find_packages(exclude=('notebooks', 'tests', 'docs'))
)
