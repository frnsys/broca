class PipeType():
    def __init__(self, name, val):
        self.name = name
        self.val = val


class _PipeTypes(type):
    _types = {}

    def __getattr__(cls, name):
        if name not in cls._types:
            cls._types[name] = PipeType(name, len(cls._types))
        return cls._types[name]


class PipeTypes(metaclass=_PipeTypes):
    pass


class Pipe():
    input = None
    output = None
    type = PipeTypes

    def __new__(cls, *args, **kwargs):
        obj = super().__new__(cls)
        obj.args = args
        obj.kwargs = kwargs

        # Build Pipe's signature
        args = ', '.join([ags for ags in [
            ', '.join(map(str, args)),
            ', '.join(['{}={}'.format(x, y) for x, y in kwargs.items()])
        ] if ags])
        obj.sig = '{}({})'.format(
            cls.__name__,
            args
        )

        return obj

    def __init__(self, *args, **kwargs):
        self.args = args
        for k, v in kwargs.items():
            setattr(self, k, v)

    def __repr__(self):
        return self.sig


class IdentityPipe(Pipe):
    """
    The identity pipe is useful for branching,
    i.e. when you need to pass an input unmodified to a pipe
    further along.
    """
    def __init__(self, pipe_type):
        self.input = pipe_type
        self.output = pipe_type

    def __call__(self, input):
        return input
