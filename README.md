# Broca
### text generation models

These are a couple text generation models with common interfaces.


## Installation

    pip install broca


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

## Contributing
If you have your own text generation approach or have a model that's missing from here,
it would be great if you could share it! The only two requirements for contribution are
keeping the model API consistent (with `train` and `speak` methods) and then an example
to go in the `examples/` folder.
