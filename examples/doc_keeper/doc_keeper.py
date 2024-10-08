from src.parrot import tasker


@tasker
class DocKeeper:

    @tasker.setup
    def setup(self, filepath, thread):
        # parse file to string
        # chunk file

        pass
