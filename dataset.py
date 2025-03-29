import torch
import numpy as np
from torch.utils.data import Dataset
import os

class CATHDataset(Dataset):
    def __init__(self, split, fold, dataset_name):
        cur_dir = os.path.dirname(os.path.realpath(__file__))  # Get current script directory
        npz_file_path = os.path.join(cur_dir, f'cache/cv_data/{dataset_name}_{split}_fold{fold}.npz')
        data = np.load(npz_file_path, allow_pickle=True)

        self.positions = data['positions']
        self.labels = data['labels']

    def __len__(self):
        return len(self.positions)

    def __getitem__(self, idx):
        position = self.positions[idx].astype(float)
        label = self.labels[idx]
        return torch.tensor(position, dtype=torch.float32), torch.tensor(label, dtype=torch.long)

if __name__ == "__main__":
    dataset = CATHDataset("test", 9, "cath_10arch_ca")
    position, label = dataset.__getitem__(1)