from collections import UserList

from torch.utils.data import DataLoader


class BatchList(UserList):
    def __init__(self, dataloaders: list[DataLoader]):
        super().__init__(dataloaders)

    def to(self, *args, **kwargs):
        return BatchList([data.to(*args, **kwargs) for data in self.data])
