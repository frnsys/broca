## Roadmap

[X] restructure library
[X] standardize APIs
[ ] add tests
[ ] add examples (though the tests could function as examples too)
[ ] write better documentation


## Overview

`broca` is a NLP library for experimenting with various approaches.

When I implement a new method, often from a paper or another source, I add it here so that it can be re-applied elsewhere.
Eventually I hope that `broca` can become a battery of experimental NLP methods which can easily be thrown at a new problem.


## Notes

`broca` is currently structured like so:

- common: misc utilities and classes reused across the whole library. Also includes shared objects.
- distance: for measuring string distance. This should probably be renamed though, since "distance" means a lot more than just string distance.
- generate: these are models which generate text but this might be confused with "generative models", so this module should be renamed, or I may move it to a separate project (`wernicke`).
- keywords: various keyword extraction methods. this module can be cleaned up/restructured.
- vectorize: there are a lot of different ways of representing text in vector space; this module is meant to contain those different methods.
- visualize: convenience stuff for visualizing output...not sure how necessary this is.
