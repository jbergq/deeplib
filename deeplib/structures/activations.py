from .storage import Storage


class Activations(Storage):
    def __init__(self, storage):
        super().__init__(storage)
