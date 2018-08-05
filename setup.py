from setuptools import find_packages, setup

setup(name='eliot',
      version='0.1',
      description='A poetry generation library.',
      packages=find_packages(),
      install_requires=['nltk', 'pronouncing'])
