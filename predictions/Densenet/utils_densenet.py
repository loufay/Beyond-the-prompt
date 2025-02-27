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
import wandb
from sklearn.metrics import roc_auc_score
from sklearn.metrics import ConfusionMatrixDisplay
from sklearn.metrics import recall_score, precision_score
from sklearn.metrics import balanced_accuracy_score
from sklearn.metrics import f1_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import matthews_corrcoef


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


# Training loop
def train(model, train_loader, val_loader, num_epochs, criterion, optimizer,scheduler, save_path):
    best_val_loss = float('inf')
    early_stop_counter = 0
    patience = 5

    for epoch in range(num_epochs):
        model.train()
        train_loss = 0
        train_preds, train_labels, train_probs = [], [], []

        for images, labels in tqdm(train_loader, desc=f"Training Epoch {epoch+1}/{num_epochs}"):
            images, labels = images.cuda(), labels.float().cuda()

            optimizer.zero_grad()
            outputs = model(images).squeeze()
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            train_prob = torch.sigmoid(outputs)
            train_pred = (train_prob > 0.5).cpu().numpy().tolist()

            train_probs.extend(train_prob.cpu().detach().numpy().tolist())
            train_preds.extend(train_pred)
            train_labels.extend(labels.cpu().numpy())

        train_loss /= len(train_loader)
        train_acc = accuracy_score(train_labels, train_preds)
        train_auc = roc_auc_score(train_labels, train_probs)

        model.eval()
        val_loss = 0
        val_preds, val_labels, val_probs = [], [], []
        
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.cuda(), labels.float().cuda()

                outputs = model(images).squeeze()
                loss = criterion(outputs, labels)

                val_loss += loss.item()
                val_prob = torch.sigmoid(outputs)
                val_pred = (val_prob > 0.5).cpu().numpy().tolist()

                val_probs.extend(val_prob.cpu().detach().numpy().tolist())
                val_preds.extend(val_pred)
                val_labels.extend(labels.cpu().numpy())


        val_loss /= len(val_loader)
        val_acc = accuracy_score(val_labels, val_preds)
        val_auc = roc_auc_score(val_labels, val_probs)

        print(f"Epoch [{epoch+1}/{num_epochs}] - Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f} - Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")
        wandb.log({"Train Loss": train_loss, "Train Accuracy": train_acc, "Validation Loss": val_loss, "Validation Accuracy": val_acc, "Train AUC": train_auc, "Validation AUC": val_auc, "epoch": epoch})
        
        scheduler.step(val_loss)
        current_lr = optimizer.param_groups[0]['lr']
        wandb.log({"learning_rate": current_lr})
        print(f"Current learning rate: {current_lr}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), os.path.join(save_path, "best_model.pth"))
            print("Model saved!")
            early_stop_counter = 0
        else:
            early_stop_counter += 1
            print(f"No improvement in validation loss. Early stopping counter: {early_stop_counter}/{patience}")


        if early_stop_counter >= patience:
            print("Early stopping triggered.")
            break


# Testing
def test(model, test_loader, criterion, df_test, bias_variables=None):
    model.eval()
    test_loss = 0
    test_preds, test_labels, test_probs = [], [], []

    with torch.no_grad():
        for images, labels in tqdm(test_loader, desc="Testing"):
            images, labels = images.cuda(), labels.float().cuda()

            outputs = model(images).squeeze()
            loss = criterion(outputs, labels)
            test_loss += loss.item()

            test_prob = torch.sigmoid(outputs).cpu().detach().numpy()
            test_pred = (test_prob > 0.5).astype(int)

            test_probs.extend(test_prob.tolist())
            test_preds.extend(test_pred)
            test_labels.extend(labels.cpu().numpy())


    test_loss /= len(test_loader)
    test_accuracy = accuracy_score(test_labels, test_preds)
    test_auc = roc_auc_score(test_labels, test_probs)

    cm = confusion_matrix(test_labels, test_preds)

    print(f"Testing Loss: {test_loss:.4f}, Accuracy: {test_accuracy:.4f}")
    no_findings_accuracy = cm[0, 0] / cm[0].sum() if cm[0].sum() > 0 else 0
    pneumonia_accuracy = cm[1, 1] / cm[1].sum() if cm[1].sum() > 0 else 0

    print(f"No Findings Accuracy: {no_findings_accuracy:.4f}, Pneumonia Accuracy: {pneumonia_accuracy:.4f}")
    wandb.log({"test_loss": test_loss, "test_accuracy": test_accuracy, "no_findings_accuracy": no_findings_accuracy, "pneumonia_accuracy": pneumonia_accuracy, "test_auc": test_auc})

    # sklearn balanced accuracy
    balanced_accuracy = balanced_accuracy_score(test_labels, test_preds)
    print(f"Balanced Accuracy: {balanced_accuracy:.4f}")
    wandb.log({"balanced_accuracy": balanced_accuracy})

    f1_score_c = f1_score(test_labels, test_preds, average='macro')
    print(f"F1 Score: {f1_score_c:.4f}")
    wandb.log({"f1_score": f1_score_c})

    # Compute recall and precision sklearn.metrics.recall_score, sklearn.metrics.precision_score
    recall = recall_score(test_labels, test_preds, average='macro')
    precision = precision_score(test_labels, test_preds, average='macro')
    print(f"Recall: {recall:.4f}, Precision: {precision:.4f}")


    # Plot confusion matrix
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["No Finding", "Pneumonia"])
    disp.plot(cmap=plt.cm.Blues)
    plt.title(f"Confusion Matrix: {wandb.run.group}_{wandb.run.name}")
    plt.tight_layout()
    plt.savefig(os.path.join(args.save_path, "test_confusion_matrix.png"))
    wandb.log({"confusion_matrix": wandb.Image(plt)})

        
    if bias_variables is not None:
        # Perform bias evaluation
        evaluate_bias(df_test, test_preds, test_labels, bias_variables)


    wandb.finish()
