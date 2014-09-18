from setuptools import setup

setup(
    name='broca',
    version='0.1.0',
    description='text generation models',
    url='https://github.com/publicscience/broca',
    author='Francis Tseng',
    author_email='f@frnsys.com',
    license='GPLv3',

    packages=['broca'],
    install_requires=[
        'nltk',
        'numpy'
    ],
)
