from __future__ import unicode_literals

import re
from broca.tokenize.keyword import gram_size, lemma_forms, spacy


def markup_terms(docs, t_docs):
    """
    Highlights each instance of the given term
    in the document. All forms of the term will be highlighted.
    """
    # Cache compiled regexes for term forms
    form_regs = {}
    highlighted = []

    for doc, t_doc in zip(docs, t_docs):
        t_doc = set(t_doc)
        t_doc = sorted(list(t_doc), key=lambda t: len(t), reverse=True) # Longest first
        toks = spacy(' '.join(t_doc), tag=True, parse=False, entity=False)
        tagged_doc = [(t.lower_.strip(), t.tag_) for t in toks if t.lower_.strip()]
        for t in t_doc:
            for term in t.split(','):
                term = term.strip()

                # Determine which forms are present for the term in the document
                if gram_size(term) == 1:
                    # Replace longer forms first so we don't replace their substrings.
                    forms = sorted(lemma_forms(term, tagged_doc), key=lambda f: len(f), reverse=True)
                else:
                    forms = [term]

                for t in forms:
                    reg = form_regs.get(t, None)
                    if reg is None:
                        # This captures 'F.D.A' if given 'FDA'
                        # yeah, it's kind of overkill
                        reg_ = '[.]?'.join(list(t))

                        # Spaces might be spaces, or they might be hyphens
                        reg_ = reg_.replace(' ', '[\s-]')

                        # Only match the term if it is not continguous with other characters.
                        # Otherwise it might be a substring of another word, which we want to
                        # ignore
                        # The last matching group is to try and ignore things which are
                        # in html tags.
                        reg = re.compile('(^|{0})({1})($|{0})(?=[^>]*(<|$))'.format('[^A-Za-z]', reg_), flags=re.IGNORECASE)
                        form_regs[t] = reg

                    if reg.findall(doc):
                        doc = reg.sub('\g<1><span class="highlight" data-term="{0}">\g<2></span>\g<3>'.format(term), doc)
                    else:
                        # If none of the term was found, try with extra alpha characters
                        # This helps if a phrase was newly learned and only assembled in
                        # its lemma form, so we may be missing the actual form it appears in.
                        reg = re.compile('(^|{0})({1}[A-Za-z]?)()(?=[^>]*(<|$))'.format('[^A-Za-z]', reg_), flags=re.IGNORECASE)
                        doc = reg.sub('\g<1><span class="highlight" data-term="{0}">\g<2></span>\g<3>'.format(term), doc)
        highlighted.append(doc)

    return highlighted
