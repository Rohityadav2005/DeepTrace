import torch
from torch.utils.data import Dataset
import cv2
import os
from pathlib import Path

class DeepfakeDataset(Dataset):
    def __init__(self, df, seq_len=30, transform=None, root_dir="C:\Users\rohit\OneDrive\Desktop\DeepFakeDetectionSystem"):
        self.df = df
        self.seq_len = seq_len
        self.transform = transform
        self.root_dir = Path(root_dir) if root_dir else Path.cwd()

    def __len__(self):
        return len(self.df) * 2
    
    def __getitem__(self, index):
        row = self.df.iloc[index // 2]

        rel_path = row["orignal_video_path"] if index % 2 == 0 else row["fake_video_path"]
        label = 1 if index % 2 == 0 else 0

        folder = self.root_dir / rel_path

        if not folder.exists():
            raise FileNotFoundError(f"Directory not found: {folder.absolute()}\n"
                                    f"Check if your CSV paths align with your project structure.")

        frame_files = sorted([f for f in os.listdir(folder) if f.endswith(('.png', '.jpg', '.jpeg'))])[:self.seq_len]

        frames = []
        for file in frame_files:
            img_path = str(folder / file)
            img = cv2.imread(img_path)
            
            if img is None:
                continue 
                
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

            if self.transform:
                img = self.transform(img)

            frames.append(img)

        if len(frames) == 0:
            raise ValueError(f"No valid images found in {folder}")

        frames = torch.stack(frames) 
        return frames, label