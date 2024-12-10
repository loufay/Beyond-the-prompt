import os
import sys
import base64
import argparse
import pandas as pd
import numpy as np
import torch
from torch import nn, optim
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, accuracy_score, ConfusionMatrixDisplay
from tqdm import tqdm
import wandb
from utils import read_image, create_wandb_run_name, calculate_subgroup_metrics, balance_dataset
current_dir = os.getcwd()
current_dir = current_dir + "/MedImageInsights"
sys.path.append(current_dir)
from MedImageInsight.medimageinsightmodel import MedImageInsight
from dataset.PneumoniaDataset import PneumoniaDataset
from dataset.PneumoniaModel import PneumoniaModel
from torch.utils.data import DataLoader
from peft import LoraConfig, get_peft_model
import warnings
warnings.filterwarnings("ignore", category=FutureWarning, module="torch.utils.checkpoint")
import wandb


parser = argparse.ArgumentParser(description="Adapter fine-tuning using LORA.")
parser.add_argument("--dataset", type=str, default="MIMIC", help="Dataset to use (MIMIC, CheXpert, VinDR)")
parser.add_argument("--save_path", type=str, default=current_dir+"/Results/", help="Path to save the results")
parser.add_argument("--only_no_finding", action="store_true", help="Filter reports for 'No Finding' samples")
parser.add_argument("--single_disease", action="store_true", help="Filter reports for single disease occurrence")
parser.add_argument("--train_data_percentage", type=float, default=1.0, help="Percentage of training data to use")
args = parser.parse_args()

#DEBUG

run_name = create_wandb_run_name(args, "lora")

# Initialize W&B
wandb.init(
    project="MedImageInsights_3",
    group=f"{args.dataset}-LORA",
    name=run_name,
)


PATH_TO_DATA = os.path.join(current_dir, "data")

if args.dataset == "MIMIC":
    data_path = os.path.join(PATH_TO_DATA, "MIMIC-v1.0-512")
    results_path = os.path.join(args.save_path, "MIMIC-v1.0-512")

elif args.dataset == "CheXpert":
    data_path = os.path.join(PATH_TO_DATA, "CheXpert-v1.0-512")
    results_path = os.path.join(args.save_path, "CheXpert-v1.0-512")

elif args.dataset == "VinDr":
    data_path = os.path.join(PATH_TO_DATA, "vindr-pcxr")
    results_path = os.path.join(args.save_path, "vindr-pcxr")


if not os.path.exists(results_path):
    os.makedirs(results_path)

df_train = pd.read_csv(os.path.join(data_path, "train.csv"))
df_val = pd.read_csv(os.path.join(data_path, "val.csv"))
df_test = pd.read_csv(os.path.join(data_path, "test.csv"))

df_train = df_train[(df_train["No Finding"] == 1) | (df_train["Pneumonia"] == 1)]
df_val = df_val[(df_val["No Finding"] == 1) | (df_val["Pneumonia"] == 1)]
df_test = df_test[(df_test["No Finding"] == 1) | (df_test["Pneumonia"] == 1)]

df_train = balance_dataset(df_train, "Pneumonia", args.train_data_percentage)
df_val = balance_dataset(df_val, "Pneumonia")
df_test = balance_dataset(df_test, "Pneumonia")


train_dataset = PneumoniaDataset(df=df_train, data_dir=PATH_TO_DATA)
val_dataset = PneumoniaDataset(df=df_val, data_dir=PATH_TO_DATA)
test_dataset = PneumoniaDataset(df=df_test, data_dir=PATH_TO_DATA)

print(f"Train dataset size: {len(train_dataset)}")
print(f"Validation dataset size: {len(val_dataset)}")
print(f"Test dataset size: {len(test_dataset)}")

# Create DataLoader
train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True, num_workers=4)
val_loader = DataLoader(val_dataset, batch_size=8, shuffle=False, num_workers=4)
test_loader = DataLoader(test_dataset, batch_size=8, shuffle=False, num_workers=4)

# Define LoRA configuration targeting attention and FFN layers
lora_config = LoraConfig(
    r=8,  # LoRA rank
    lora_alpha=16,  # Scaling factor
    target_modules=[
        "window_attn.fn.qkv", 
        "window_attn.fn.proj", 
        "ffn.fn.net.fc1", 
        "ffn.fn.net.fc2"
    ],  # Target attention and FFN weights
    lora_dropout=0.1,
    bias="none",  # Don't modify biases
    task_type="vision"
)

classifier = MedImageInsight(
    model_dir=os.path.join(current_dir, "MedImageInsight/2024.09.27"),
    vision_model_name="medimageinsigt-v1.0.0.pt",
    language_model_name="language_model.pth"
)

classifier.load_model()
classifier.model.to(classifier.device)

# Apply LoRA to the image encoder
classifier.model.image_encoder = get_peft_model(classifier.model.image_encoder, lora_config)
model = PneumoniaModel(classifier.model).to(classifier.device)


# Freeze base model parameters except for LoRA parameters
for name, param in model.base_model.named_parameters():
    if 'lora' not in name:
        param.requires_grad = False

# Ensure classifier parameters are trainable
for param in model.classifier.parameters():
    param.requires_grad = True

# Collect trainable parameters (LoRA and classifier)
trainable_params = [
    {'params': [p for n, p in model.base_model.named_parameters() if 'lora' in n]},
    {'params': model.classifier.parameters()}
]

# Define optimizer
optimizer = optim.AdamW(trainable_params, lr=1e-4)
# Define loss and optimizer
criterion = nn.BCEWithLogitsLoss()
# Define learning rate scheduler
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=2, verbose=True)

# Move model to appropriate device
classifier.model.to(classifier.device)


# Define hyperparameters
num_epochs = 100
patience = 5  # Early stopping patience
best_val_loss = float('inf')
early_stop_counter = 0
model_save_path = results_path + "/best_model.pth"

for epoch in range(num_epochs):
    classifier.model.train()
    running_loss = 0.0

    # For accuracy computation
    train_preds = []
    train_labels = []

    # Training loop
    for images, labels in tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}", unit="batch"):
        # Preprocess images
        image_list = [classifier.decode_base64_image(img_base64) for img_base64 in images]
        image_tensors = torch.stack([classifier.preprocess(img) for img in image_list]).to(classifier.device)
        labels = labels.float().to(classifier.device)

        # Forward pass
        logits = model(image_tensors)
        loss = criterion(logits, labels)

        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

        # Collect predictions and labels for accuracy
        train_preds.extend((logits.squeeze() > 0.5).cpu().numpy())  # Threshold for binary predictions
        train_labels.extend(labels.cpu().numpy())       

    train_loss = running_loss / len(train_loader)
    train_accuracy = accuracy_score(train_labels, train_preds)
    print(f"Epoch [{epoch+1}/{num_epochs}], Training Loss: {train_loss:.4f}, Accuracy: {train_accuracy:.4f}")
    wandb.log({"train_loss": train_loss, "train_accuracy": train_accuracy})

    # Validation loop
    classifier.model.eval()
    val_running_loss = 0.0

    # For accuracy computation
    val_preds = []
    val_labels = []

    with torch.no_grad():
        for images, labels in val_loader:
            # Preprocess images
            image_list = [classifier.decode_base64_image(img_base64) for img_base64 in images]
            image_tensors = torch.stack([classifier.preprocess(img) for img in image_list]).to(classifier.device)
            labels = labels.float().to(classifier.device)

            # Forward pass
            logits = model(image_tensors)
            val_loss = criterion(logits, labels)

            val_running_loss += val_loss.item()

            # Collect predictions and labels for accuracy
            val_preds.extend((logits.squeeze() > 0.5).cpu().numpy())
            val_labels.extend(labels.cpu().numpy())

    val_loss = val_running_loss / len(val_loader)
    val_accuracy = accuracy_score(val_labels, val_preds)
    print(f"Epoch [{epoch+1}/{num_epochs}], Validation Loss: {val_loss:.4f}, Accuracy: {val_accuracy:.4f}")
    wandb.log({"val_loss": val_loss, "val_accuracy": val_accuracy})

    # Scheduler step
    scheduler.step(val_loss)
    current_lr = optimizer.param_groups[0]['lr']
    wandb.log({"learning_rate": current_lr})
    print(f"Current learning rate: {current_lr}")

    # Check for best validation loss and save model
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        torch.save(classifier.model.state_dict(), model_save_path)
        print(f"Model saved with validation loss: {best_val_loss:.4f}")
        early_stop_counter = 0  # Reset early stopping counter
    else:
        early_stop_counter += 1
        print(f"No improvement in validation loss. Early stopping counter: {early_stop_counter}/{patience}")

    # Early stopping
    if early_stop_counter >= patience:
        print("Early stopping triggered.")
        break

# Testing after training
classifier.model.load_state_dict(torch.load(model_save_path))  # Load best model
classifier.model.eval()
test_running_loss = 0.0

# For accuracy computation
test_preds = []
test_labels = []

with torch.no_grad():
    for images, labels in tqdm(test_loader, desc="Testing", unit="batch"):
        # Preprocess images
        image_list = [classifier.decode_base64_image(img_base64) for img_base64 in images]
        image_tensors = torch.stack([classifier.preprocess(img) for img in image_list]).to(classifier.device)
        labels = labels.float().to(classifier.device)

        # Forward pass
        logits = model(image_tensors)
        test_loss = criterion(logits, labels)

        test_running_loss += test_loss.item()

        # Collect predictions and labels for accuracy
        test_preds.extend((logits.squeeze() > 0.5).cpu().numpy())
        test_labels.extend(labels.cpu().numpy())

test_loss = test_running_loss / len(test_loader)
test_accuracy = accuracy_score(test_labels, test_preds)
cm = confusion_matrix(test_labels, test_preds)

print(f"Testing Loss: {test_loss:.4f}, Accuracy: {test_accuracy:.4f}")
no_findings_accuracy = cm[0, 0] / cm[0].sum() if cm[0].sum() > 0 else 0
pneumonia_accuracy = cm[1, 1] / cm[1].sum() if cm[1].sum() > 0 else 0
print(f"No Findings Accuracy: {no_findings_accuracy:.4f}, Pneumonia Accuracy: {pneumonia_accuracy:.4f}")
wandb.log({"test_loss": test_loss, "test_accuracy": test_accuracy, "no_findings_accuracy": no_findings_accuracy, "pneumonia_accuracy": pneumonia_accuracy})

# Plot confusion matrix
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["No Finding", "Pneumonia"])
disp.plot(cmap=plt.cm.Blues)
plt.title(f"Confusion Matrix: {wandb.run.group}_{wandb.run.name}")
plt.tight_layout()
plt.savefig(os.path.join(args.save_path, "test_confusion_matrix.png"))
wandb.log({"confusion_matrix": wandb.Image(plt)})
wandb.finish()
