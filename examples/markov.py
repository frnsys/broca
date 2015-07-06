from broca.generate import Markov
from examples import docs

m = Markov()

print('Training on {0} docs...'.format(len(docs)))
m.train(docs)

for i in range(20):
    print('\n---\n')
    print(m.speak())
