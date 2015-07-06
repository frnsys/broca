from broca.generate import Madlib
from examples import docs

m = Madlib()

print('Training on {0} docs...'.format(len(docs)))
m.train(docs)

for i in range(20):
    print('\n---\n')
    print(m.speak())
