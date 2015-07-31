# Broca
### Various useful NLP algos and utilities

There is some Python 2 support scattered throughout but the library has not been fully tested against it.

**This library is in development -- APIs may change.**


## Overview

`broca` is a NLP library for experimenting with various approaches. So everything in this library is somewhat experimental and meant for rapid prototyping of NLP methods.

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
- `pipeline`: for easily chaining `broca` classes into pipelines - useful for rapid prototyping


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

You can use `broca`'s module conventionally, or you can take advantage of its pipelines:

```python
from broca.pipeline import Pipeline
from broca.preprocess import Cleaner, HTMLCleaner
from broca.vectorize import BoW, DCS

p = Pipeline(
        HTMLCleaner(),
        Cleaner(),
        BoW()
    )

vecs = p(docs)
```

Pipelines allow you to chain `broca`'s objects and easily swap them out.

You can also build multi-pipelines to try out a variety of pipelines simultaneously:

```python
p = Pipeline(
        HTMLCleaner(),
        Cleaner(),
        [BoW(), DCS()]
    )

vecs1, vecs2 = p(docs)
```

You can also nest pipelines and multi-pipelines:

```python
clean = Pipeline(
            HTMLCleaner(),
            Cleaner(),
        )

vectr_pipeline = Pipeline(
    clean,
    [BoW(), DCS()]
)

vecs1, vecs2 = p(docs)
```

Pipelines are validated upon creation to ensure that the outputs and inputs of adjacent components ("pipes") are compatible.

### Freezing pipes

By default, pipelines are frozen - that is, each pipe's output memoized to disk based on the inputs it receives. If the input changes or the pipe's `__call__` method is redefined, its output will be recomputed; otherwise, it will be loaded from disk. This means you can easily swap out components in a pipeline without needing to redundantly recompute parts which are not affected.

You can disable this behavior for a pipeline by specifying `freeze=False`:

```python
p = Pipeline(
        HTMLCleaner(),
        Cleaner(),
        freeze=False
    )
```

You can force the recomputation of an entire pipeline by specifying `refresh=True`:

```python
p = Pipeline(
        HTMLCleaner(),
        Cleaner(),
        refresh=True
    )
```

### Implementing a pipe

Implementing your own pipeline component is easy. Just define a class which inherits from `broca.pipeline.Pipe` and define its `__call__` method and `input` and `output` class attributes, which should be from `Pipe.type`.

The call method must take only two arguments: `self` and then the input from the preceding pipe. If there are parameters to be specified, they should be handled in the pipe's `__init__` method.

```python
from broca.pipeline import Pipe

class MyPipe(Pipe):
    input = Pipe.type.docs
    output = Pipe.type.vecs

    def __init__(self, some_param):
        self.some_param = some_param

    def __call__(self, docs):
        # do something with docs to get vectors
        vecs = make_vecs_func(docs, self.some_param)
        return vecs
```


## Examples

There are a few usage examples in the `examples` directory.


## Tests

Unit tests can be run using `nose`:

    $ nosetests tests
