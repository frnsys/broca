from itertools import product
from broca.pipeline.cryo import Cryo


class Pipeline():
    def __init__(self, *pipes, **kwargs):
        self.freeze = kwargs.get('freeze', True)
        self.refresh = kwargs.get('refresh', False)
        self.cryo = Cryo(refresh=self.refresh)

        # If any of the pipes is a list, we are building multiple pipelines
        if any(isinstance(p, list) for p in pipes):
            # Coerce all pipes to lists
            pipes = [p if isinstance(p, list) else [p] for p in pipes]

            # Build each pipeline
            self.pipelines = [Pipeline(*pipes_) for pipes_ in product(*pipes)]

        else:
            self.pipes = pipes

            # Validate the pipeline
            for p_out, p_in in zip(pipes, pipes[1:]):
                if p_out.output != p_in.input:
                    raise Exception('Incompatible: pipe <{}> outputs <{}>, pipe <{}> requires input of <{}>.'.format(
                        type(p_out).__name__, p_out.output,
                        type(p_in).__name__, p_in.input
                    ))

    def __call__(self, input):
        if hasattr(self, 'pipelines'):
            return [p(input) for p in self.pipelines]
        else:
            for pipe in self.pipes:
                output = self.cryo(pipe, input) if self.freeze else pipe(input)
                input = output
            return output

    def __repr__(self):
        if hasattr(self, 'pipelines'):
            return 'MultiPipeline: {}'.format(' || '.join([str(p) for p in self.pipelines]))
        else:
            return ' -> '.join([str(p) for p in self.pipes])


class PipeType():
    tokens = 'tokens'
    assetid_doc = 'assetid_doc' # type == dict,  {asset_id : body_text }
    assetid_vec = 'assetid_vec' # type == dict,  {asset_id : vec }
    docs = 'docs'
    vecs = 'vecs'
    sim_mat = 'sim_mat'


class Pipe():
    input = None
    output = None
    type = PipeType

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

    def __repr__(self):
        return self.sig
