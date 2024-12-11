import os
import sys
import argparse
import pandas as pd
import numpy as np
import torch
from torch import nn, optim
from torch.utils.data import DataLoader
from torchvision import models, transforms
from sklearn.metrics import accuracy_score, f1_score, recall_score, precision_score, balanced_accuracy_score, confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
from tqdm import tqdm
from utils_densenet import read_image, balance_dataset
from dataset import PneumoniaDataset
import wandb
current_dir = os.getcwd()
current_dir = current_dir + "/MedImageInsights"
sys.path.append(current_dir)


# Define transformations
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Argument parser
parser = argparse.ArgumentParser(description="End-to-end training for pneumonia prediction.")
parser.add_argument("--dataset", type=str, default="MIMIC", help="Dataset to use (MIMIC, CheXpert, VinDR)")
parser.add_argument("--save_path", type=str, default="./results", help="Path to save results.")
parser.add_argument("--batch_size", type=int, default=32, help="Batch size.")
parser.add_argument("--epochs", type=int, default=100, help="Number of training epochs.")
parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate.")
parser.add_argument("--train_data_percentage", type=float, default=1, help="Percentage of training data to use.")
args = parser.parse_args()

wandb.init(
    project="MedImageInsights_3",
    group=f"{args.dataset}-DenseNet",
    name="DenseNet",
)


PATH_TO_DATA = os.path.join(current_dir, "data")

if args.dataset == "MIMIC":
    data_path = os.path.join(PATH_TO_DATA, "MIMIC-v1.0-512")
    results_path = os.path.join(args.save_path, "MIMIC-v1.0-512")



if not os.path.exists(results_path):
    os.makedirs(results_path)

train_df = pd.read_csv(os.path.join(data_path, "train.csv"))
val_df = pd.read_csv(os.path.join(data_path, "val.csv"))
test_df = pd.read_csv(os.path.join(data_path, "test.csv"))

train_df = train_df[(train_df["No Finding"] == 1) | (train_df["Pneumonia"] == 1)]
val_df = val_df[(val_df["No Finding"] == 1) | (val_df["Pneumonia"] == 1)]
test_df = test_df[(test_df["No Finding"] == 1) | (test_df["Pneumonia"] == 1)]

# balance the dataset
train_df = balance_dataset(train_df, "Pneumonia", args.train_data_percentage)
val_df = balance_dataset(val_df, "Pneumonia")
test_df = balance_dataset(test_df, "Pneumonia")


train_dataset = PneumoniaDataset(train_df, PATH_TO_DATA, transform=transform)
val_dataset = PneumoniaDataset(val_df, PATH_TO_DATA, transform=transform)
test_dataset = PneumoniaDataset(test_df, PATH_TO_DATA, transform=transform)

train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=4)
val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4)
test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4)

# Model definition
model = models.densenet121(pretrained=True)
model.classifier = nn.Linear(model.classifier.in_features, 1)
model = model.cuda()


# Loss function and optimizer
criterion = nn.BCEWithLogitsLoss()
optimizer = optim.Adam(model.parameters(), lr=args.lr)

# Learning rate scheduler
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3, verbose=True)


# Training loop
def train(model, train_loader, val_loader, num_epochs, criterion, optimizer, save_path):
    best_val_loss = float('inf')
    early_stop_counter = 0
    patience = 5

    for epoch in range(num_epochs):
        model.train()
        train_loss = 0
        train_preds, train_labels = [], []

        for images, labels in tqdm(train_loader, desc=f"Training Epoch {epoch+1}/{num_epochs}"):
            images, labels = images.cuda(), labels.float().cuda()

            optimizer.zero_grad()
            outputs = model(images).squeeze()
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            train_preds.extend((outputs > 0).cpu().numpy())
            train_labels.extend(labels.cpu().numpy())

        train_loss /= len(train_loader)
        train_acc = accuracy_score(train_labels, train_preds)

        model.eval()
        val_loss = 0
        val_preds, val_labels = [], []
        
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.cuda(), labels.float().cuda()

                outputs = model(images).squeeze()
                loss = criterion(outputs, labels)

                val_loss += loss.item()
                val_preds.extend((outputs > 0).cpu().numpy())
                val_labels.extend(labels.cpu().numpy())

        val_loss /= len(val_loader)
        val_acc = accuracy_score(val_labels, val_preds)

        print(f"Epoch [{epoch+1}/{num_epochs}] - Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f} - Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")
        wandb.log({"Train Loss": train_loss, "Train Accuracy": train_acc, "Validation Loss": val_loss, "Validation Accuracy": val_acc})
        
        scheduler.step(val_loss)

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

# Train the model
if not os.path.exists(args.save_path):
    os.makedirs(args.save_path)

train(model, train_loader, val_loader, args.epochs, criterion, optimizer, args.save_path)

# Testing
def test(model, test_loader, criterion):
    model.eval()
    test_loss = 0
    test_preds, test_labels = [], []

    with torch.no_grad():
        for images, labels in tqdm(test_loader, desc="Testing"):
            images, labels = images.cuda(), labels.float().cuda()

            outputs = model(images).squeeze()
            loss = criterion(outputs, labels)
            test_loss += loss.item()

            test_preds.extend((outputs > 0).cpu().numpy())
            test_labels.extend(labels.cpu().numpy())

    test_loss /= len(test_loader)
    test_accuracy = accuracy_score(test_labels, test_preds)
    cm = confusion_matrix(test_labels, test_preds)

    print(f"Testing Loss: {test_loss:.4f}, Accuracy: {test_accuracy:.4f}")
    no_findings_accuracy = cm[0, 0] / cm[0].sum() if cm[0].sum() > 0 else 0
    pneumonia_accuracy = cm[1, 1] / cm[1].sum() if cm[1].sum() > 0 else 0

    print(f"No Findings Accuracy: {no_findings_accuracy:.4f}, Pneumonia Accuracy: {pneumonia_accuracy:.4f}")
    wandb.log({"test_loss": test_loss, "test_accuracy": test_accuracy, "no_findings_accuracy": no_findings_accuracy, "pneumonia_accuracy": pneumonia_accuracy})

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
    wandb.finish()

    

# Load best model
model.load_state_dict(torch.load(os.path.join(args.save_path, "best_model.pth")))

# Evaluate on test set
test(model, test_loader, criterion)
