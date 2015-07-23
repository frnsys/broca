# Broca
### Various useful NLP algos and utilities

There is some Python 2 support scattered throughout but the library has not been fully tested against it.

**This library is in development -- APIs may change.**

## Overview

`broca` is a NLP library for experimenting with various approaches.

When I implement a new method, often from a paper or another source, I add it here so that it can be re-applied elsewhere.
Eventually I hope that `broca` can become a battery of experimental NLP methods which can easily be thrown at a new problem.

`broca` is structured like so:

- `common`: misc utilities and classes reused across the whole library. Also includes shared objects.
- `distance`: for measuring string distance. This should probably be renamed though, since "distance" means a lot more than just string distance.
- `tokenize`: various tokenization methods
    - `keyword`: keyword-based tokenization methods (i.e. keyword extraction methods)
- `vectorize`: various ways of representing documents as vectors
- `similarity`: various ways of computing similarity
    - `term`: for computing similarity between two terms
    - `doc`: for computing similarity matrices for sets of documents
- `preprocess`: for preprocessing text, i.e. cleaning
- `knowledge`: tools for preparing or incorporating external knowledge sources, such as Wikipedia or IDF on auxiliary corpora
- `visualize`: convenience stuff for visualizing output
- `pipeline`: for easily chaining `broca` classes into pipelines - useful for rapidly iterating


## Installation

`broca` is available through pypi, but the library is under active development, so it's recommended to install via git:

    $ pip install git+ssh://git@github.com/ftzeng/broca.git

Or, if adding to a `requirements.txt`, add the line:

    git+ssh://git@github.com/ftzeng/broca.git

If developing, you can clone the repo and from within the repo directory, install via `pip`:

    $ pip install --editable .

Your installed version will be aliased directly from the repo directory, so changes are always immediately accessible.

You also need to install the `spacy` library's data:

    $ python -m spacy.en.download


## Usage

(to do)


## Examples

You can get a sense of the keyword extractor quality by running the `examples/keywords.py` script.

(to do)


## Tests

Unit tests can be run using `nose`:

    $ nosetests tests
