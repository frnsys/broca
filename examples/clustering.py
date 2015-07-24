"""
This example demonstrates using broca with scikit-learn for clustering.

A multi-pipeline is assembled to try four different bag-of-words vectorizers, each
with a different tokenizer, and then compare their K-Means clustering results.
"""
from sklearn import metrics
from sklearn.cluster import KMeans
from broca.vectorize import BoW
from broca.pipeline import Pipeline
from broca.preprocess import Cleaner
from broca.tokenize.keyword import Overkill, RAKE, POS
from examples import load_data

# A dataset of news article clusters
data = load_data('10E.json')

# Prep ground truth labelings
docs = []
true = []
for i, e in enumerate(data):
    for a in e['articles']:
        docs.append(a['body'])
        true.append(i)

# Prep Pipeline
pipeline = Pipeline(Cleaner(), [
                        BoW(),
                        BoW(tokenizer=Overkill),
                        BoW(tokenizer=RAKE),
                        BoW(tokenizer=POS)
                    ])

# Run the pipeline and get the results
for i, vecs in enumerate(pipeline(docs)):
    print('\n----------\n')
    print('Pipeline: ', pipeline.pipelines[i])

    # Feed the resulting vectors into KMeans
    model = KMeans(n_clusters=10)
    pred = model.fit_predict(vecs)

    print('Completeness', metrics.completeness_score(true, pred))
    print('Homogeneity', metrics.homogeneity_score(true, pred))
    print('Adjusted Mutual Info', metrics.adjusted_mutual_info_score(true, pred))
    print('Adjusted Rand', metrics.adjusted_rand_score(true, pred))
