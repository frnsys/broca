# Broca
### Various useful NLP algos and utilities

There is some Python 2 support scattered throughout but the library has not been fully tested against it.

**This library is in development -- APIs may change and features may be unstable.**


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

You also need to install the `spacy` and `nltk` libraries' data:

    $ python -m spacy.en.download
    $ python -m nltk.downloader all


## Usage

You can use `broca`'s module conventionally, or you can take advantage of its pipelines:

```python
from broca import Pipeline
from broca.preprocess import BasicCleaner, HTMLCleaner
from broca.vectorize import BoWVectorizer, DCSVectorizer

p = Pipeline(
        HTMLCleaner(),
        BasicCleaner(),
        BoWVectorizer()
    )

vecs = p(docs)
```

Pipelines allow you to chain `broca`'s objects and easily swap them out.

Pipelines are validated upon creation to ensure that the outputs and inputs of adjacent components ("pipes") are compatible.

### Multi-pipelines

You can also build multi-pipelines to try out a variety of pipelines simultaneously:

```python
p = Pipeline(
        HTMLCleaner(),
        BasicCleaner(),
        [BoWVectorizer(), DCSVectorizer()]
    )

vecs1, vecs2 = p(docs)
```

This results in two pipelines which are run simultaneously when `p(docs)` is executed:

- `HTMLCleaner() -> BasicCleaner() -> BoWVectorizer()`
- `HTMLCleaner() -> BasicCleaner() -> DCSVectorizer()`

### Nesting pipelines

You can also nest pipelines and multi-pipelines:

```python
clean = Pipeline(
            HTMLCleaner(),
            BasicCleaner(),
        )

vectr_pipeline = Pipeline(
    clean,
    [BoWVectorizer(), DCSVectorizer()]
)

vecs1, vecs2 = p(docs)
```

### Branching

Pipes can support input from multiple pipes or output to multiple pipes simultaneously.

Where multi-pipelines create distinct and separate pipelines, a branching pipeline is a singular pipeline where its inputs get mapped at branching segments and reduced afterwards.

Branches are specified as tuples.

Here's an example:

```python
class A(Pipe):
    input = Pipe.type.vals
    output = Pipe.type.vals
    def __call__(self, vals):
        return [v+1 for v in vals]

class B(Pipe):
    input = Pipe.type.vals
    output = Pipe.type.vals
    def __call__(self, vals):
        return [v+2 for v in vals]

class C(Pipe):
    input = Pipe.type.vals
    output = Pipe.type.vals
    def __call__(self, vals):
        return [v+3 for v in vals]

class D(Pipe):
    input = Pipe.type.vals
    output = Pipe.type.vals
    def __call__(self, vals):
        return [v+4 for v in vals]

class E(Pipe):
    input = (Pipe.type.vals, Pipe.type.vals, Pipe.type.vals)
    output = Pipe.type.vals
    def __call__(self, vals1, vals2, vals3):
        return [sum([v1,v2,v3]) for v1,v2,v3 in zip(vals1,vals2,vals3)]

branching_pipeline = Pipeline(
        A(),
        (B(), C(), D()) # A branching segment
        (B(), C(), D()) # Another branching segment
        E()             # Reduced
)

branching_pipeline([1,2,3,4])
# [24,27,30,33]
```

Whatever follows a branching segment must accept multiple inputs - this could be a single Pipe or another branching segment of equal size.

Alternatively, the `A` Pipe in the example could have had its output defined as a tuple:

```python
class A(Pipe):
    input = Pipe.type.vals
    output = (Pipe.type.vals, Pipe.type.vals, Pipe.type.vals)
    def __call__(self, vals):
        return [v+1 for v in vals], [v+2 for v in vals], [v+3 for v in vals]

# Running the above with A defined as such returns [27, 30, 33, 36] instead.
```

### Freezing pipes

By default, pipelines are frozen - that is, each pipe's output memoized to disk based on the inputs it receives. If the input changes or the pipe's `__call__` method is redefined, its output will be recomputed; otherwise, it will be loaded from disk. This means you can easily swap out components in a pipeline without needing to redundantly recompute parts which are not affected.

You can disable this behavior for a pipeline by specifying `freeze=False`:

```python
p = Pipeline(
        HTMLCleaner(),
        BasicCleaner(),
        freeze=False
    )
```

You can force the recomputation of an entire pipeline by specifying `refresh=True`:

```python
p = Pipeline(
        HTMLCleaner(),
        BasicCleaner(),
        refresh=True
    )
```

### Implementing a pipe

Implementing your own pipeline component is easy. Just define a class which inherits from `broca.pipeline.Pipe` and define its `__call__` method and `input` and `output` class attributes, which should be from `Pipe.type`.

The call method must take only two arguments: `self` and then the input from the preceding pipe. If there are parameters to be specified, they should be handled in the pipe's `__init__` method.

```python
from broca import Pipe

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

The default `__init__` method saves the initialization `args` in `self.args` and `kwargs` as properties by their key names, so you won't need to implement `__init__` if you only need it to pass arguments to `__call__`.

You can use anything for your input and output pipe types, e.g. `Pipe.type.foo` or `Pipe.type.hello_there`. They are dynamically generated as needed.

### The Identity Pipe

Sometimes you need a pipe to pass along input unmodified.

For example, the `WikipediaSimilarity` pipe takes in as input `(Pipe.type.docs, Pipe.type.tokens)`. You want to pass docs and tokenized versions of those docs.

This can be accomplished with branching and the `IdentityPipe`, which requires you specify the input pipe type:

```python

docs = [
    'I am a cat',
    'I have a lion'
]

p = Pipeline(
    (IdentityPipe(Pipe.type.docs), OverkillTokenizer()),
    WikipediaSimilarity()
)

```


## Examples

There are a few usage examples in the `examples` directory.


## Tests

Unit tests can be run using `nose`:

    $ nosetests tests
