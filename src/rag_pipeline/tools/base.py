class Tool:
    name = None
    description = None

    def run(self, query):
        raise NotImplementedError
