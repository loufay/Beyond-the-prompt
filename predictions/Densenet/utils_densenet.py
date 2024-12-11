import os
import sys
import argparse
import pandas as pd
import numpy as np
import torch
from torch import nn, optim
from torch.utils.data import DataLoader
from torchvision import models, transforms
from sklearn.metrics import accuracy_score, f1_score, recall_score, precision_score, balanced_accuracy_score, confusion_matrix
import matplotlib.pyplot as plt
from tqdm import tqdm

# Define image loading function
def read_image(img_path):
    from PIL import Image
    return Image.open(img_path).convert('RGB')


# Balance the dataset
def balance_dataset(df, disease = "Pneumonia", percentage = 1, vindr_samples = False):
    # get value count of df.disease and balance based on lowest value
    value_count = df[disease].value_counts()
    if not vindr_samples:
        minority_count = int(value_count.min()*percentage)
    else:
        vindr_split = {0.01: 6, 0.1: 74, 0.5:372, 0.8: 594, 1.0: 744}
        minority_count = int(vindr_split[percentage]/2)
        
    df_class_0 = df[df[disease] == 0].sample(minority_count, random_state=42)
    df_class_1 = df[df[disease] == 1].sample(minority_count, random_state=42)
    df_balanced = pd.concat([df_class_0, df_class_1])
    return df_balanced