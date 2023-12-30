from torch.utils.data import Dataset, DataLoader
import torch 

class TextDataset(Dataset):
    def __init__(self, encodings):
        self.encodings = encodings

    def __getitem__(self, idx):  
        return torch.tensor(self.encodings['input_ids'][idx][:-1]), torch.tensor(self.encodings['input_ids'][idx][1:]) 

    def __len__(self):
        return len(self.encodings)