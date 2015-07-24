from sklearn.linear_model import LinearRegression
from broca.vectorize import BoW
from broca.pipeline import Pipeline
from broca.preprocess import Cleaner

fake_data = [
    ('This is a very interesting comment!', 100),
    ('My comment is interesting to many people!', 120),
    ('This is a very toxic comment', -100),
    ('My behavior is very toxic', -80),
    ('What I have to say is quite interesting!', 113),
    ('I have something extremely toxic to say', -96)
]
docs, y  = list(zip(*fake_data))

pipeline = Pipeline(Cleaner(), BoW())

# Train model
vecs = pipeline(docs)
model = LinearRegression()
model.fit(vecs, y)

# Test model
test_docs = [
    'Another very interesting comment',
    'Another very toxic comment'
]
test_vecs = pipeline(test_docs)
pred = model.predict(test_vecs)

for i, doc in enumerate(test_docs):
    print(doc)
    print('Scored: ', pred[i])
