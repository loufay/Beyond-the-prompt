import torch
from torch.utils.data import Dataset
from PIL import Image
import base64
import os
import io

class PneumoniaDataset(Dataset):
    def __init__(self, df, data_dir, disease="Pneumonia"):
        """
        Args:
            df (DataFrame): DataFrame containing 'Path', 'Pneumonia', and 'No Finding' columns.
            data_dir (str): Base directory containing image files.
        """
        self.df = df
        self.data_dir = data_dir
        self.disease = disease

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        # Get row from DataFrame
        row = self.df.iloc[idx]
        
        # Construct image path
        img_path = self.data_dir + row["Path"]
        
        # Load image
        image = Image.open(img_path).convert("RGB")
        
        # Convert to base64
        buffer = io.BytesIO()
        image.save(buffer, format="JPEG")
        img_base64 = base64.encodebytes(buffer.getvalue()).decode("utf-8")

        # Determine label
        if row[self.disease] == 1:
            label = 1  # Positive
        elif row["No Finding"] == 1:
            label = 0  # Negative
        else:
            raise ValueError(f"Row {idx} has no valid label.")

        return img_base64, label
