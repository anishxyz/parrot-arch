from src.parrot import tasker


@tasker
class DocKeeper:
    """
    Orchestrating tasker
    """

    @tasker.setup
    def setup(self, config):
        # parse file to string
        # chunk file

        pass

    @tasker.run
    def run(self, filepath, thread):
        # parse file to string
        # chunk file
        # summarize doc

        pass


@tasker
class DocEditor:
    """
    Document editor
    """

    @tasker.setup
    def setup(self, config):
        # parse file to string
        # chunk file

        pass

    @tasker.run
    def run(self, chunk, instruction):
        # parse file to string
        # chunk file
        # summarize doc

        pass