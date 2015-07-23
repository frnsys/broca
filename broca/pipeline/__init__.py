class Pipeline():
    def __init__(self, *pipes):
        self.pipes = pipes

        # Validate the pipeline
        for p_out, p_in in zip(pipes, pipes[1:]):
            if p_out.output != p_in.input:
                raise Exception('Pipe {} outputs {}, pipe {} requires input of {}.'.format(
                    type(p_out).__name__, p_out.output,
                    type(p_in).__name__, p_in.input
                ))

    def __call__(self, input, debug=False):
        # Setting debug=True turns the pipeline
        # into a generator which yields intermediary output
        for pipe in self.pipes:
            output = pipe(input)
            if debug:
                yield output
            input = output
        return output


class PipeType():
    tokens = 0
    docs = 1
    vecs = 2
    sim_mat = 3
