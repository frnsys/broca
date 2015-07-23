# Broca
### Various useful NLP algos and utilities

There is some Python 2 support scattered throughout but the library has not been fully tested against it.

**This library is in development -- APIs may change.**

to do: clean up this readme :)


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
