import numpy as np
from sklearn.neighbors import KNeighborsClassifier
import os
import sys
sys.path.append(os.path.join(os.getcwd(), "MedImageInsight"))
from MedImageInsight.medimageinsightmodel import MedImageInsight



# KNN model
def create_knn_model(k_neighbors, weights):
    if weights == "distance":
        return KNeighborsClassifier(n_neighbors=k_neighbors, metric="euclidean", weights="distance")
    elif weights == "dotproduct":
        return KNeighborsClassifier(n_neighbors=k_neighbors, metric="cosine", weights="distance")
    else:
        return KNeighborsClassifier(n_neighbors=k_neighbors, metric="euclidean")


def get_medimageinsight_classifier():
    
    classifier = MedImageInsight(
        model_dir=os.path.join(os.getcwd(), "MedImageInsights/MedImageInsight/2024.09.27"),
        vision_model_name="medimageinsigt-v1.0.0.pt",
        language_model_name="language_model.pth"
    )

    classifier.load_model()
    classifier.model.to(classifier.device)
    return classifier

    