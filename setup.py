import os
from setuptools import setup

REQUIREMENTS = os.path.join(os.path.dirname(__file__), 'requirements.txt')
REQUIREMENTS = open(REQUIREMENTS, 'r').read().splitlines()

setup(
    name='broca',
    version='0.3.1',
    description='rapid nlp prototyping',
    url='https://github.com/ftzeng/broca',
    author='Francis Tseng',
    author_email='f@frnsys.com',
    license='MIT',
    packages=['broca'],
    zip_safe=True,
    package_data={'': ['broca/data']},
    include_package_data=True,
    install_requires=REQUIREMENTS,
    tests_require=['nose'],
)
