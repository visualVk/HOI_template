from torch.utils.data import Dataset
import torch
class RandomDataset(Dataset):
    def __init__(self, size, length):
        self.len = length
        self.data = torch.randn(length, size)
        self.label = torch.mean(self.data, dim=-1)

    def __getitem__(self, index):
        return self.data[index], self.label[index]

    def __len__(self):
        return self.len