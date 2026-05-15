import torch
from torch.utils.data import Dataset
import cv2
import os
from ultralytics.data.augment import LetterBox 

class PairedFoggyDataset(Dataset):
    def __init__(self, clean_root, fog_root, label_root, img_size=640):
        self.clean_files = sorted(os.listdir(clean_root))
        self.clean_root = clean_root
        self.fog_root = fog_root 
        self.label_root = label_root
        self.img_size = img_size

    def __len__(self):
        return len(self.clean_files)

    def __getitem__(self, idx):
        file_name = self.clean_files[idx]

        img_clean = cv2.imread(os.path.join(self.clean_root, file_name))
        img_fog = cv2.imread(os.path.join(self.fog_root, file_name))

        label_file = file_name.replace('.jpg', '.txt') 
        img_clean, ratio, pad = LetterBox(self.img_size, auto=False)(image=img_clean)
        img_fog, _, _ = LetterBox(self.img_size, auto=False)(image=img_fog)

        img_clean = torch.from_numpy(img_clean[:, :, ::-1]).permute(2, 0, 1).float() / 255.0
        img_fog = torch.from_numpy(img_fog[:, :, ::-1]).permute(2, 0, 1).float() / 255.0

        return {
            'img_clean': img_clean,
            'img_fog': img_fog,
            'labels': labels, 
            'img_name': file_name
        }