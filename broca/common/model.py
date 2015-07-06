import pickle

class Model():
    def save(self):
        with open(self.filepath, 'wb') as dump:
            pickle.dump(self.__dict__, dump)

    def load(self):
        with open(self.filepath, 'rb') as dump:
            me = pickle.load(dump)
        self.__dict__.update(me)
