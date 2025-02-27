import os
import sys
import argparse
import pandas as pd
import numpy as np
import torch
from torch import nn, optim
from torch.utils.data import DataLoader
from torchvision import models, transforms
from sklearn.metrics import accuracy_score, roc_auc_score, balanced_accuracy_score, f1_score, recall_score, precision_score, confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
from tqdm import tqdm
import wandb

current_dir = os.getcwd()
current_dir = current_dir + "/MedImageInsights"
sys.path.append(current_dir)
sys.path.append(current_dir+"/predictions")
from utils_densenet import read_image, balance_dataset, train, test
from dataset import PneumoniaDataset
from predictions.utils import evaluate_bias, create_wandb_run_name
from config import parse_arguments, get_dataset_config
from data_loader import load_data


def main():
    args = parse_arguments()
    dataset_config = get_dataset_config(args.dataset)
    bias_variables = dataset_config.get("bias_variables", None)

    # Initialize W&B
    base_run_name = create_wandb_run_name(args, "zeroshot")
    wandb.init(
    project=args.wandb_project,
    group=f"{args.dataset}-DenseNet",
    name=args.dataset+"_DenseNet_VinDR_split_"+str(args.train_data_percentage),
)

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])


    # Load and balance datasets
    data_path = os.path.join(os.getcwd(), "MedImageInsights/data", dataset_config["data_path"])
    df_train, df_val, df_test = load_data(data_path, args.disease)
    df_test_balanced = balance_dataset(df_test, args.disease)

    print(f'Train data: {len(df_train)}')
    print(f'Validation data: {len(df_val)}')
    print(f"Test samples: {len(df_test_balanced)}")

    # Model definition
    model = models.densenet121(pretrained=True)
    model.classifier = nn.Linear(model.classifier.in_features, 1)
    model = model.cuda()


    # Loss function and optimizer
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-4)

    # Learning rate scheduler
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3, verbose=True)

    data_path = os.path.join(os.getcwd(), "MedImageInsights/data")
    # Data loaders
    train_dataset = PneumoniaDataset(df_train, data_path, transform)
    val_dataset = PneumoniaDataset(df_val, data_path, transform)
    test_dataset = PneumoniaDataset(df_test_balanced, data_path, transform)

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False, num_workers=4)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False, num_workers=4)

        # Train the model
    if not os.path.exists(args.save_path):
        os.makedirs(args.save_path)
    epochs = 1
    train(model, train_loader, val_loader, epochs, criterion, optimizer, scheduler, args.save_path)


    # Load best model
    model.load_state_dict(torch.load(os.path.join(args.save_path, "best_model.pth")))

    # Evaluate on test set
    test(model, test_loader, criterion, df_test_balanced, bias_variables=bias_variables)


if __name__ == "__main__":
    main()


