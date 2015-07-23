from broca.common.model import Model


class Generator(Model):
    """
    Generators should inherit from this class
    """
    def train(self, docs):
        """
        Args:
            - docs -> a list or a generator of strings

        Returns: array of feature vectors
        """
        raise NotImplementedError

    def speak(self):
        raise NotImplementedError
