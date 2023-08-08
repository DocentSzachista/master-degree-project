
import torch
from torch.utils.data import Dataset


class NoisedDataset(Dataset):
    def __init__(self, data_list, target_list, transform = None) -> None:
        self.data = data_list
        self.targets = target_list
        self.transform = transform
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        sample = {'data': self.transform(self.data[idx]), 'target': self.targets[idx]}
        return sample