import pickle


class Model():
    def save(self, filepath):
        with open(filepath, 'wb') as dump:
            pickle.dump(self.__dict__, dump)

    def load(self, filepath):
        with open(filepath, 'rb') as dump:
            me = pickle.load(dump)
        self.__dict__.update(me)
