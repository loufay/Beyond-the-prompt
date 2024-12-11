import numpy as np
import torch
from torch import nn, optim
from torch.utils.data import DataLoader
from torchvision import models, transforms
from sklearn.metrics import accuracy_score, f1_score, recall_score, precision_score, balanced_accuracy_score, confusion_matrix
import matplotlib.pyplot as plt
from tqdm import tqdm
import os
from utils_densenet import read_image
# Dataset class
class PneumoniaDataset(torch.utils.data.Dataset):
    def __init__(self, dataframe, data_dir, transform=None):
        self.dataframe = dataframe
        self.data_dir = data_dir
        self.transform = transform

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, idx):
        row = self.dataframe.iloc[idx]
        img_path = self.data_dir + row["Path"]

        image = read_image(img_path)  # Implement read_image function for loading images

        if self.transform:
            image = self.transform(image)

                # Determine label
        if row["Pneumonia"] == 1:
            label = 1  # Positive
        elif row["No Finding"] == 1:
            label = 0  # Negative
        else:
            raise ValueError(f"Row {idx} has no valid label.")


        return image, label