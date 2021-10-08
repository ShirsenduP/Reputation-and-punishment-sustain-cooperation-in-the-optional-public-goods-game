from setuptools import setup, find_packages

setup(
   name='opgar',
   version='1.0',
   description='Optional Public Good game And Reputation',
   author='Shirsendu Podder',
   author_email='ucabpod@ucl.ac.uk',
   packages=['opgar'],  #same as name
   install_requires=['numpy', 'pandas', 'tqdm', 'networkx'], #external packages as dependencies
)