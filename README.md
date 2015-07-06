# Broca
### Various useful NLP algos and utilities

There is some Python 2 support scattered throughout but the library has not been fully tested against it.

This library is in development -- APIs may change.

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

## The Models
There are only two models at the moment.

### Markov
`Markov` is a simple Markov chain generator. It learns the probabilities of a token A
following a token B and then builds a chain of tokens based on those probabilities.

### Madlib
`Madlib` is a model similar to a [context-free grammar](http://www.rednoise.org/pdal/index.php?n=Main.Grammars).
It learns part-of-speech patterns and builds a vocabulary from a set of documents. Then
it just randomly swaps in new words in according to their part-of-speech.


## Usage
The models share a very simple common interface:

    # Train the model on some documents:
    model.train(docs)

    # Generate some speech with the model:
    model.speak()

Each model has a few of its own configuration options. There's not much code so you can
refer to the inline documentation for more info on those options.


## Examples
These examples run off a sample (n=10000) from a dataset of all the plot descriptions of IMDB.

    # Madlib
    python examples/madlib.py

    # Markov
    python examples/markov.py

The `Madlib` example takes awhile to train :\


## Tests

You can get a sense of the keyword extractor quality by running the `examples/keywords.py` script.

Unit tests can be run using `nose`:

    $ nosetests tests

## Contributing
If you have your own text generation approach or have a model that's missing from here,
it would be great if you could share it! The only two requirements for contribution are
keeping the model API consistent (with `train` and `speak` methods) and then an example
to go in the `examples/` folder.
