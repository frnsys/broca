"""
Disambiguated Core Semantics (DCS) [EXPERIMENTAL]
----------------------------------

As described in:

    Tingting Wei, Yonghe Lu, Huiyou Chang, Qiang Zhou, Xianyu Bao (2015).
    A semantic approach for text clustering using WordNet and lexical chains.
    <http://www.sciencedirect.com/science/article/pii/S0957417414006472>

Overview:

    DCS disambiguates word senses (for nouns, verbs, adverbs, and adjectives) by
    selecting the word sense most appropriate given the other word senses in a document,
    "most appropriate" based on semantic similarity (calculated using a combination of
    Wu-Palmer similarity and a novel "implicit" semantic similarity, see code for details).

    Lexical chains are constructed out of these senses ("concepts"), which are then scored
    according to the frequencies of a chain's concepts in each document. The n highest-scoring lexical
    chains are selected to represent the document.

    The aggregate set of qualifying concepts for all documents defines the feature space
    for vector representations.

    Here "concept" is synonymous with "synset".
    "Concept" is how the paper refers to synsets, so that's used here.

Limitations:

    - some words may not be in WordNet, e.g. entities or neologisms
    - some semantic relationships are not represented in WordNet
    - there's a lot going on here, it can be quite slow

Notes:

    - This code is more or less a sketch, it could be optimized ~
    - I have had limited success using this representation for clustering

- Francis
"""

import math
import numpy as np
from nltk.corpus import stopwords
from nltk.corpus import wordnet as wn
from scipy.sparse.csgraph import connected_components
from broca.distance.sift4 import sift4
from broca.vectorize import Vectorizer
from broca.common.util import penn_to_wordnet
from broca.common.shared import spacy


stops = stopwords.words('english')


class DCS(Vectorizer):
    def __init__(self, alpha=1.5, relation_weights=[0.8, 0.5, 0.3], n_chains=10):
        self.alpha = 1.5
        self.relation_weights = relation_weights
        self.n_chains = n_chains

        # Cache concept => description
        # and (c1, c2) => similarity
        self.descriptions = {}
        self.concept_sims = {}


    def vectorize(self, docs):
        """
        Vectorizes a list of documents using their DCS representations.
        """
        doc_core_sems, all_concepts = self._extract_core_semantics(docs)

        shape = (len(docs), len(all_concepts))
        vecs = np.zeros(shape)
        for i, core_sems in enumerate(doc_core_sems):
            for con, weight in core_sems:
                j = all_concepts.index(con)
                vecs[i,j] = weight

        # Normalize
        return vecs/np.max(vecs)


    def _process_doc(self, doc):
        """
        Applies DCS to a document to extract its core concepts and their weights.
        """
        # Prep
        doc = doc.lower()
        tagged_tokens = [(t, penn_to_wordnet(t.tag_)) for t in spacy(doc, tag=True, parse=False, entity=False)]
        tokens = [t for t, tag in tagged_tokens]
        term_concept_map = self._disambiguate_doc(tagged_tokens)
        concept_weights = self._weight_concepts(tokens, term_concept_map)

        # Compute core semantics
        lexical_chains = self._lexical_chains(doc, term_concept_map)
        core_semantics = self._core_semantics(lexical_chains, concept_weights)
        core_concepts = [c for chain in core_semantics for c in chain]

        return [(con, concept_weights[con]) for con in core_concepts]


    def _disambiguate_doc(self, tagged_tokens):
        """
        Takes a list of tagged tokens, representing a document,
        in the form:

            [(token, tag), ...]

        And returns a mapping of terms to their disambiguated concepts (synsets).
        """

        # Group tokens by PoS
        pos_groups = {pos: [] for pos in [wn.NOUN, wn.VERB, wn.ADJ, wn.ADV]}
        for tok, tag in tagged_tokens:
            if tag in pos_groups:
                pos_groups[tag].append(tok)

        #print(pos_groups)

        # Map of final term -> concept mappings
        map = {}
        for tag, toks in pos_groups.items():
            map.update(self._disambiguate_pos(toks, tag))

        #nice_map = {k: map[k].lemma_names() for k in map.keys()}
        #print(json.dumps(nice_map, indent=4, sort_keys=True))

        return map


    def _disambiguate_pos(self, terms, pos):
        """
        Disambiguates a list of tokens of a given PoS.
        """
        # Map the terms to candidate concepts
        # Consider only the top 3 most common senses
        candidate_map = {term: wn.synsets(term, pos=pos)[:3] for term in terms}

        # Filter to unique concepts
        concepts = set(c for cons in candidate_map.values() for c in cons)

        # Back to list for consistent ordering
        concepts = list(concepts)
        sim_mat = self._similarity_matrix(concepts)

        # Final map of terms to their disambiguated concepts
        map = {}

        # This is terrible
        # For each term, select the candidate concept
        # which has the maximum aggregate similarity score against
        # all other candidate concepts of all other terms sharing the same PoS
        for term, cons in candidate_map.items():
            # Some words may not be in WordNet
            # and thus have no candidate concepts, so skip
            if not cons:
                continue
            scores = []
            for con in cons:
                i = concepts.index(con)
                scores_ = []
                for term_, cons_ in candidate_map.items():
                    # Some words may not be in WordNet
                    # and thus have no candidate concepts, so skip
                    if term == term_ or not cons_:
                        continue
                    cons_idx = [concepts.index(c) for c in cons_]
                    top_sim = max(sim_mat[i,cons_idx])
                    scores_.append(top_sim)
                scores.append(sum(scores_))
            best_idx = np.argmax(scores)
            map[term] = cons[best_idx]

        return map


    def _similarity_matrix(self, concepts):
        """
        Computes a semantic similarity matrix for a set of concepts.
        """
        n_cons = len(concepts)
        sim_mat = np.zeros((n_cons, n_cons))
        for i, c1 in enumerate(concepts):
            for j, c2 in enumerate(concepts):
                # Just build the lower triangle
                if i >= j:
                    sim_mat[i,j] = self._semsim(c1, c2) if i != j else 1.
        return sim_mat + sim_mat.T - np.diag(sim_mat.diagonal())


    def _semsim(self, c1, c2):
        """
        Computes the semantic similarity between two concepts.

        The semantic similarity is a combination of two sem sims:

            1. An "explicit" sem sim metric, that is, one which is directly
            encoded in the WordNet graph. Here it is just Wu-Palmer similarity.

            2. An "implicit" sem sim metric. See `_imp_semsim`.

        Note we can't use the NLTK Wu-Palmer similarity implementation because we need to
        incorporate the implicit sem sim, but it's fairly straightforward --
        leaning on <http://www.nltk.org/_modules/nltk/corpus/reader/wordnet.html#Synset.wup_similarity>,
        see that for more info. Though...the formula in the paper includes an extra term in the denominator,
        which is wrong, so we leave it out.
        """
        if c1 == c2:
            return 1.

        if (c1, c2) in self.concept_sims:
            return self.concept_sims[(c1, c2)]

        elif (c2, c1) in self.concept_sims:
            return self.concept_sims[(c2, c1)]

        else:
            need_root = c1._needs_root()
            subsumers = c1.lowest_common_hypernyms(c2, simulate_root=need_root)

            if not subsumers:
                # For relationships not in WordNet, fallback on just implicit semsim.
                return self._imp_semsim(c1, c2)

            subsumer = subsumers[0]
            depth = subsumer.max_depth() + 1
            len1 = c1.shortest_path_distance(subsumer, simulate_root=need_root)
            len2 = c2.shortest_path_distance(subsumer, simulate_root=need_root)

            if len1 is None or len2 is None:
                # See above
                return self._imp_semsim(c1, c2)

            len1 += depth
            len2 += depth

            imp_score = self._imp_semsim(c1, c2)

            sim = (2.*depth + imp_score)/(len1 + len2 + imp_score)
            self.concept_sims[(c1, c2)] = sim
            return sim


    def _imp_semsim(self, c1, c2):
        """
        The paper's implicit semantic similarity metric
        involves iteratively computing string overlaps;
        this is a modification where we instead use
        inverse Sift4 distance (a fast approximation of Levenshtein distance).

        Frankly ~ I don't know if this is an appropriate
        substitute, so I'll have to play around with this and see.
        """

        desc1 = self._description(c1)
        desc2 = self._description(c2)

        raw_sim = 1/(sift4(desc1, desc2) + 1)
        return math.log(raw_sim + 1)


    def _core_semantics(self, lex_chains, concept_weights):
        """
        Returns the n representative lexical chains for a document.
        """
        chain_scores = [self._score_chain(lex_chain, adj_submat, concept_weights) for lex_chain, adj_submat in lex_chains]
        scored_chains = zip(lex_chains, chain_scores)
        scored_chains = sorted(scored_chains, key=lambda x: x[1], reverse=True)

        thresh = (self.alpha/len(lex_chains)) * sum(chain_scores)
        return [chain for (chain, adj_mat), score in scored_chains if score >= thresh][:self.n_chains]


    def _extract_core_semantics(self, docs):
        """
        Extracts core semantics for a list of documents, returning them along with
        a list of all the concepts represented.
        """
        all_concepts = []
        doc_core_sems = []
        for doc in docs:
            core_sems = self._process_doc(doc)
            doc_core_sems.append(core_sems)
            all_concepts += [con for con, weight in core_sems]
        return doc_core_sems, list(set(all_concepts))


    def _lexical_chains(self, doc, term_concept_map):
        """
        Builds lexical chains, as an adjacency matrix,
        using a disambiguated term-concept map.
        """
        concepts = list({c for c in term_concept_map.values()})

        # Build an adjacency matrix for the graph
        # Using the encoding:
        # 1 = identity/synonymy, 2 = hypernymy/hyponymy, 3 = meronymy, 0 = no edge
        n_cons = len(concepts)
        adj_mat = np.zeros((n_cons, n_cons))

        for i, c in enumerate(concepts):
            # TO DO can only do i >= j since the graph is undirected
            for j, c_ in enumerate(concepts):
                edge = 0
                if c == c_:
                    edge = 1
                # TO DO when should simulate root be True?
                elif c_ in c._shortest_hypernym_paths(simulate_root=False).keys():
                    edge = 2
                elif c in c_._shortest_hypernym_paths(simulate_root=False).keys():
                    edge = 2
                elif c_ in c.member_meronyms() + c.part_meronyms() + c.substance_meronyms():
                    edge = 3
                elif c in c_.member_meronyms() + c_.part_meronyms() + c_.substance_meronyms():
                    edge = 3

                adj_mat[i,j] = edge

        # Group connected concepts by labels
        concept_labels = connected_components(adj_mat, directed=False)[1]
        lexical_chains = [([], []) for i in range(max(concept_labels) + 1)]
        for i, concept in enumerate(concepts):
            label = concept_labels[i]
            lexical_chains[label][0].append(concept)
            lexical_chains[label][1].append(i)

        # Return the lexical chains as (concept list, adjacency sub-matrix) tuples
        return [(chain, adj_mat[indices][:,indices]) for chain, indices in lexical_chains]


    def _score_chain(self, lexical_chain, adj_submat, concept_weights):
        """
        Computes the score for a lexical chain.
        """
        scores = []

        # Compute scores for concepts in the chain
        for i, c in enumerate(lexical_chain):
            score = concept_weights[c] * self.relation_weights[0]
            rel_scores = []
            for j, c_ in enumerate(lexical_chain):
                if adj_submat[i,j] == 2:
                    rel_scores.append(self.relation_weights[1] * concept_weights[c_])

                elif adj_submat[i,j] == 3:
                    rel_scores.append(self.relation_weights[2] * concept_weights[c_])

            scores.append(score + sum(rel_scores))

        # The chain's score is just the sum of its concepts' scores
        return sum(scores)


    def _weight_concepts(self, tokens, term_concept_map):
        """
        Calculates weights for concepts in a document.

        This is just the frequency of terms which map to a concept.
        """

        weights = {c: 0 for c in term_concept_map.values()}
        for t in tokens:
            # Skip terms that aren't one of the PoS we used
            if t not in term_concept_map:
                continue
            con = term_concept_map[t]
            weights[con] += 1

        # TO DO paper doesn't mention normalizing these weights...should we?
        return weights


    def _description(self, concept):
        """
        Returns a "description" of a concept,
        as defined in the paper.

        The paper describes the description as a string,
        so this is a slight modification where we instead represent
        the definition as a list of tokens.
        """
        if concept not in self.descriptions:
            lemmas = concept.lemma_names()
            gloss = self._gloss(concept)
            glosses = [self._gloss(rel) for rel in self._related(concept)]
            raw_desc = ' '.join(lemmas + [gloss] + glosses)
            desc = [w for w in raw_desc.split() if w not in stops]
            self.descriptions[concept] = desc
        return self.descriptions[concept]


    def _gloss(self, concept):
        """
        The concatenation of a concept's definition and its examples.
        """
        return  ' '.join([concept.definition()] + concept.examples())


    def _related(self, concept):
        """
        Returns related concepts for a concept.
        """
        return concept.hypernyms() + \
                concept.hyponyms() + \
                concept.member_meronyms() + \
                concept.substance_meronyms() + \
                concept.part_meronyms() + \
                concept.member_holonyms() + \
                concept.substance_holonyms() + \
                concept.part_holonyms() + \
                concept.attributes() + \
                concept.also_sees() + \
                concept.similar_tos()
