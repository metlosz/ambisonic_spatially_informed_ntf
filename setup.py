from setuptools import setup
from os import path


with open(path.join('README.md'), encoding='utf-8') as file:
    long_description = file.read()

setup(name='asintf',
      author='Mateusz Guzik',
      description='',
      long_description=long_description,
      long_description_content_type='text/markdown',
      url='https://github.com/metlosz/ambisonic_spatially_informed_ntf_private',
      packages=['asintf'])
