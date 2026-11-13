import cv2
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

class MeterDataset(Dataset):
    def __init__(self, folder):
        self.folder = folder
        self.df = pd.read_csv(f"{folder}/labels.csv", dtype={"label": str})

    def __len__(self):
        return len(self.df)          

    def __getitem__(self, i):
        row = self.df.iloc[i]
        path = f"{self.folder}/{row.filename}"
        img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        img = img.astype(np.float32) / 255.0     
        t = torch.from_numpy(img)                
        t = t.unsqueeze(0)                      
        label = torch.tensor([int(c) for c in row.label])   
        return t, label
     
from torch.utils.data import DataLoader, random_split


if __name__ == "__main__":
    ds = MeterDataset("data/meter_images")
    train_ds, val_ds = random_split(ds, [4000, 1000])

    train_loader = DataLoader(train_ds, batch_size=64, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=64)

    x, y = next(iter(train_loader))
    print(x.shape, y.shape)   # [64, 1, 32, 160]  [64, 5]
    print(y[0])               # compare with the first image's label in labels.csv