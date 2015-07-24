from itertools import product


class Pipeline():
    def __init__(self, *pipes):
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
                output = pipe(input)
                input = output
            return output


    def __repr__(self):
        if hasattr(self, 'pipelines'):
            return 'Multi-Pipeline: {}'.format(' || '.join([str(p) for p in self.pipelines]))
        else:
            return ' -> '.join([type(p).__name__ for p in self.pipes])


class PipeType():
    tokens = 'tokens'
    docs = 'docs'
    vecs = 'vecs'
    sim_mat = 'sim_mat'