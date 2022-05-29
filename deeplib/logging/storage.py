from torchvision.utils import make_grid


class Storage:
    """
    Class for holding logged data from model.
    """

    def __init__(self, storage={}) -> None:
        self.storage = storage

    def items(self):
        return self.storage.items()

    def get(self, key=""):
        return [data for name, data in self.storage.items() if key in name]

    def __getitem__(self, key):
        return self.storage[key]

    def __setitem__(self, key, value):
        self.storage[key] = value

    def __contains__(self, key):
        return key in self.storage

    def __len__(self):
        return len(self.storage)

    def __iter__(self):
        return iter(self.storage)
