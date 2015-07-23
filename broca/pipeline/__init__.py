class Pipeline():
    def __init__(self, *pipes):
        self.pipes = pipes

        # Validate the pipeline
        for p_out, p_in in zip(pipes, pipes[1:]):
            if p_out.output != p_in.input:
                raise Exception('Incompatible: pipe <{}> outputs <{}>, pipe <{}> requires input of <{}>.'.format(
                    type(p_out).__name__, p_out.output,
                    type(p_in).__name__, p_in.input
                ))

    def __call__(self, input):
        for pipe in self.pipes:
            output = pipe(input)
            input = output
        return output


class PipeType():
    tokens = 'tokens'
    docs = 'docs'
    vecs = 'vecs'
    sim_mat = 'sim_mat'
