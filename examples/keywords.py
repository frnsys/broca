"""
Runs all keyword extractors and displays their outputs.
"""

import os
import pkgutil
import importlib
from broca import keywords

# Load all the keyword extractor modules.
package = os.path.dirname(keywords.__file__)
modules = [name for _, name, _ in pkgutil.iter_modules([package])]
extractors = {module: importlib.import_module('{0}.{1}'.format(keywords.__name__, module)) for module in modules}

# http://www.nytimes.com/interactive/2015/04/14/dining/field-guide-to-the-sandwich.html
docs = [
    '''
    Here is a celebration of the sandwich's diversity in the United States, an attempt to bring order to the wild multiplicity of its forms.
    ''',
    '''
    But first: What is a sandwich? The United States Department of Agriculture declares: "Product must contain at least 35 percent cooked meat and no more than 50 percent bread." But a sandwich does not require meat! Merriam-Webster is slightly more helpful: "two or more slices of bread or a split roll having a filling in between."
    ''',
    '''
    For the purposes of this field guide, we have laid down parameters. A hamburger is a marvelous sandwich, but it is one deserving of its own guide. The same holds for hot dogs, and for tacos and burritos, which in 2006, in the case known as Panera v. Qdoba, a Massachusetts judge declared were not sandwiches at all. Open-faced sandwiches are not sandwiches. Gyros and shawarmas are not sandwiches. The bread that encases them is neither split nor hinged, but wrapped.
    '''
]

# Eh should get other (human) input on this too
expected = {
    'sandwich',
    'united states',
    'united states department of agriculture',
    'merriam-webster',
    'bread',
    'hamburger',
    'hot dogs',
    'tacos',
    'burritos',
    'field guide',
    'panera',
    'qdoba',
    'massachusetts',
    'sandwiches',
    'open-faced sandwiches',
    'gyros',
    'shawarmas'
}

for name, mod in extractors.items():
    print('\n{0}\n~~~~~~~~~~~'.format(name))
    keywords = mod.extract_keywords(docs)

    # Flatten
    keywords = [k for kws in keywords for k in kws]

    # Convert to set for Jaccard score.
    keywords = set([k.lower() for k in keywords])
    score = len(expected.intersection(keywords))/len(expected.union(keywords))

    print('Score:\t\t{0}'.format(score))
    print('Keywords:\t{0}'.format(keywords))
    print('Missing:\t{0}'.format(expected.difference(keywords)))
    print('Extras:\t\t{0}'.format(keywords.difference(expected)))
