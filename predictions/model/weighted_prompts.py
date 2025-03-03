import os
import sys
import torch
import numpy as np
import pandas as pd
import wandb
import matplotlib.pyplot as plt
import seaborn as sns
from config import parse_arguments, get_dataset_config
from data_loader import load_data, prepare_samples
from metrics import evaluate_model
from sklearn.metrics import confusion_matrix, roc_auc_score, accuracy_score, matthews_corrcoef
from sklearn.metrics.pairwise import cosine_similarity
from tqdm import tqdm
from PIL import Image

current_dir = os.getcwd()
current_dir = current_dir + "/MedImageInsights"
sys.path.append(current_dir)

from MedImageInsight.medimageinsightmodel import MedImageInsight
from utils import (
    read_image, 
    zero_shot_prediction, 
    create_wandb_run_name, 
    augment_image_to_base64, 
    select_confident_samples, balance_dataset
)
from model.model import get_medimageinsight_classifier
from model.WeightedPromptEnsemble import WeightedPromptEnsemble, ZeroShotClassifier, prepare_samples




def extract_weighted_text_embeddings(args):
  
    PATH_TO_DATA = os.path.join(os.getcwd(), "MedImageInsights/data")
    data_path = os.path.join(PATH_TO_DATA, "MIMIC-v1.0-512")



    # Load text embeddings 46 per class
    if args.text_processing == "weighted_all":
        path_to_embeddings = f'{current_dir}/data/text_embeddings/embeddings_92_no_finding.npy'
    elif args.text_processing == "weighted_prompts_only":
        path_to_embeddings = f'{current_dir}/data/text_embeddings/embeddings_92_no_finding_template.npy'
    elif args.text_processing == "weighted_reports_only":
        path_to_embeddings = f'{current_dir}/data/text_embeddings/embeddings_92_no_finding_report.npy'



    embeddings = np.load(path_to_embeddings)
    embedding_length = embeddings.shape[0]//2
    no_finding_templates = torch.tensor(embeddings[:embedding_length])
    disease_templates = torch.tensor(embeddings[embedding_length:])

    # Load image embeddings
    df_train = pd.read_csv(os.path.join(data_path, "train.csv"))
    df_train = df_train[(df_train["No Finding"] == 1) | (df_train[args.disease] == 1)]
    df_train_balanced = balance_dataset(df_train, args.disease, 1, True)

    df_test = pd.read_csv(os.path.join(data_path, "test.csv"))
    df_test = df_test[(df_test["No Finding"] == 1) | (df_test[args.disease] == 1)]
    df_test_balanced = balance_dataset(df_test, args.disease)

    print(f"Train: {len(df_train_balanced)} samples")
    print(f"Test: {len(df_test_balanced)} samples")

    train_samples = prepare_samples(df_train_balanced, df_train_balanced.columns[-1024:], args.disease)
    train_image_embeddings = torch.tensor(train_samples["features"])

    test_samples = prepare_samples(df_test_balanced, df_test_balanced.columns[-1024:], args.disease)
    test_image_embeddings = torch.tensor(test_samples["features"])

    # Initialize model
    model = ZeroShotClassifier(num_prompts_no_finding=no_finding_templates.shape[0],
                                num_prompts_pneumonia=disease_templates.shape[0],
                            embedding_dim=disease_templates.shape[1])

    # initialize model weights equally for all prompts
    model.initialize_weights(no_finding_templates, disease_templates)

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-3)

    # Early stopping parameters
    tolerance = 1e-5
    previous_loss = float('inf')
    smallest_loss = float('inf')
    patience = 50
    counter = 0

    init_ensemble_no_finding, init_ensemble_disease = model.ensemble(no_finding_templates, disease_templates)
    init_ensemble_no_finding = init_ensemble_no_finding.detach().clone()
    init_ensemble_disease = init_ensemble_disease.detach().clone()
    init_alpha_no_finding = model.ensemble.alpha_no_finding
    init_alpha_pneumonia = model.ensemble.alpha_pneumonia

    # Training loop for learning weights
    for epoch in tqdm(range(1000), desc="Training Progress"):  # Added tqdm for progress tracking
        optimizer.zero_grad()
        probabilities = model(train_image_embeddings, no_finding_templates, disease_templates)
        loss = model.entropy_loss(probabilities)
        #loss = model.kl_divergence(probabilities)
        loss.backward()
        optimizer.step()

        tqdm.write(f'Epoch {epoch + 1}, Loss: {loss.item():.6f}')  # Print loss neatly with tqdm

        # save ensemble weights with smallest loss
        if loss.item() < smallest_loss:
            smallest_loss = loss.item()
            smallest_ensemble_no_finding, smallest_ensemble_disease = model.ensemble(no_finding_templates, disease_templates)
            smallest_ensemble_no_finding = smallest_ensemble_no_finding.detach().clone()
            smallest_ensemble_disease = smallest_ensemble_disease.detach().clone()
            smallest_epoch = epoch
            smallest_alpha_no_finding = model.ensemble.alpha_no_finding
            smallest_alpha_pneumonia = model.ensemble.alpha_pneumonia
        
        # Early stopping condition
        if abs(previous_loss - loss.item()) < tolerance:
            counter += 1
            if counter >= patience:
                print("Early stopping triggered.")
                break
        else:
            counter = 0  # Reset if improvement observed

        previous_loss = loss.item()

        # compute auc
        auc = roc_auc_score(train_samples["labels"], probabilities[:, 1].detach().numpy())
        cm = confusion_matrix(train_samples["labels"], np.argmax(probabilities.detach().numpy(), axis=1))
        acc = np.trace(cm) / np.sum(cm)
        tpr = cm[1, 1] / np.sum(cm[1])
        tnr = cm[0, 0] / np.sum(cm[0])
        tqdm.write(f"AUC: {auc:.4f}, Accuracy: {acc:.4f}, TNR: {tnr:.4f}, TPR: {tpr:.4f}, Loss: {loss.item():.6f}")


    # Freeze the learned weights
    for param in model.ensemble.parameters():
        param.requires_grad = False

    # Generate fixed ensemble embeddings
    final_ensemble_no_finding, final_ensemble_disease= model.ensemble(no_finding_templates, disease_templates)
    final_ensemble_no_finding = final_ensemble_no_finding.detach().clone()
    final_ensemble_disease = final_ensemble_disease.detach().clone()

    # expand dimensions
    final_ensemble_no_finding = final_ensemble_no_finding.unsqueeze(0)
    final_ensemble_disease = final_ensemble_disease.unsqueeze(0)

    final_ensemble = torch.cat([final_ensemble_no_finding, final_ensemble_disease], dim=0)


    return final_ensemble