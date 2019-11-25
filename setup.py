#!/usr/bin/env python

from setuptools import setup, find_packages

long_description='''
DNN-Builder is a library to build neural network architectures from building blocks,
 for example custom versions of DenseNet or ResNet for specific application.

This library is only compatible with Python 3.x.
'''

setup(name='DNN-Builder',
      version='0.0.1',
      description='Neural network building blocks in Keras',
      long_description=long_description,
      author='Matias Valdenegro-Toro',
      author_email='matias.valdenegro@gmail.com',
      url='https://github.com/mvaldenegro/dnn-builder',
      download_url='https://github.com/mvaldenegro/dnn-builder/releases',
      license='LGPLv3',
      install_requires=['keras>=2.2.0', 'numpy'],
      packages=find_packages()
     )