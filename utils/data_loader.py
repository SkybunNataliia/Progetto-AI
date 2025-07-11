import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader

class BikeSharingDataset(Dataset):
    """
    Dataset custom PyTorch per dati con sliding window.
    Input: sequenza di lunghezza seq_len con features multiple.
    Target: valore count dell'ora successiva.
    """

    def __init__(self, csv_path, seq_len=24):
        """
        csv_path: path del file csv preprocessato (train/val/test)
        seq_len: lunghezza della finestra temporale (es. 24 ore)
        """
        self.seq_len = seq_len
        self.df = pd.read_csv(csv_path)
        
        self.features = self.df.drop(columns=['count']).values
        self.targets = self.df['count'].values

    def __len__(self):
        return len(self.features) - self.seq_len
    
    def __getitem__(self, idx):
        x = self.features[idx : idx + self.seq_len]
        y = self.targets[idx + self.seq_len]
        return torch.tensor(x, dtype=torch.float32), torch.tensor(y, dtype=torch.float32)

def get_dataloaders(train_csv, val_csv, test_csv, batch_size=32, seq_len=24):
    """
    Crea DataLoader per train, val, test.
    """
    train_dataset = BikeSharingDataset(train_csv, seq_len)
    val_dataset = BikeSharingDataset(val_csv, seq_len)
    test_dataset = BikeSharingDataset(test_csv, seq_len)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader, test_loader


if __name__ == "__main__":
    train_loader, val_loader, test_loader = get_dataloaders(
        "data/processed/train.csv",
        "data/processed/val.csv",
        "data/processed/test.csv",
        batch_size=16,
        seq_len=24,
    )
    print(f"Numero batch train: {len(train_loader)}")