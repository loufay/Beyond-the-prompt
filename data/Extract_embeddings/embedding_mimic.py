from sklearn.metrics import confusion_matrix, roc_auc_score, accuracy_score, precision_score, recall_score, f1_score
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
import sys
sys.path.append("/mnt/data2/datasets_lfay/MedImageInsights")
from medimageinsightmodel import MedImageInsight
import base64
import os
from sklearn.metrics import accuracy_score
from utils import read_image
import torch 

chunk_size = 10  
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


## LOAD DATASET
df_train = pd.read_csv("/mnt/data2/datasets_lfay/MedImageInsights/data/MIMIC-v1.0-512/extracted_labels.csv")
datasets = [("train", df_train)]

## COMPUTE IMAGE EMBEDDINGS
model = MedImageInsight(
    model_dir="/mnt/data2/datasets_lfay/MedImageInsights/2024.09.27",
    vision_model_name="medimageinsigt-v1.0.0.pt",
    language_model_name="language_model.pth"
)
model.load_model()



for df in datasets:
    images = []
    embeddings = {}

    for i, row in df[1].iterrows():
        path_to_mimic_img = row["filename"].replace(".txt", "") + "/" + row["dicom_id"] + ".jpg"
        image = base64.encodebytes(read_image(path_to_mimic_img)).decode("utf-8")
        images.append(image)
        with torch.no_grad():
            embedding = model.encode([image])
        embeddings[path_to_mimic_img] = embedding["image_embeddings"].flatten()
        
        if i%1000 == 0:
            print(f"Computed image embeddings for {i} images")
            # save embeddings dict
            embeddings_df = pd.DataFrame.from_dict(embeddings, orient="index")
            embeddings_df.to_csv(f"/mnt/data2/datasets_lfay/MedImageInsights/data/MIMIC-v1.0-512/{df[0]}_embeddings.csv")
            
    # save embeddings dict
    embeddings_df = pd.DataFrame.from_dict(embeddings, orient="index")

    embeddings_df.to_csv(f"/mnt/data2/datasets_lfay/MedImageInsights/data/MIMIC-v1.0-512/{df[0]}_embeddings.csv")

    
    print(f"Computed image embeddings for {df[0]} dataset")
    print("Number of image embeddings:", len(embeddings_df))





