import os
import sys
import torch
import numpy as np
import pandas as pd
import wandb
from config import parse_arguments, get_dataset_config
from data_loader import load_data, prepare_samples
from metrics import evaluate_model, plot_confusion_matrix
from model.adapter_training import create_data_loader
current_dir = os.getcwd()
current_dir = current_dir + "/MedImageInsights"
sys.path.append(current_dir)

from MedImageInsight.medimageinsightmodel import MedImageInsight
from dataset.PneumoniaDataset import PneumoniaDataset
from dataset.PneumoniaModel import PneumoniaModel
from torch.utils.data import DataLoader
from peft import LoraConfig, get_peft_model
from utils import create_wandb_run_name, balance_dataset, evaluate_bias
from model.model import get_medimageinsight_classifier
from tqdm import tqdm
from sklearn.metrics import accuracy_score


def main():
    # Parse arguments and get dataset-specific configurations
    args = parse_arguments()
    dataset_config = get_dataset_config(args.dataset)
    bias_variables = dataset_config.get("bias_variables", None)

    # Initialize W&B
    run_name = create_wandb_run_name(args, "lora")
    wandb.init(
        project=args.wandb_project,
        group=f"{args.dataset}-LORA",
        name=run_name,
    )
    output_dir = os.path.join(args.save_path, f"{args.dataset}-LORA", f"{wandb.run.group}_{wandb.run.name}")
    os.makedirs(output_dir, exist_ok=True)

    # Load and balance datasets
    data_path = os.path.join(os.getcwd(), "MedImageInsights/data")
    df_train, df_val, df_test = load_data(data_path + "/" + dataset_config["data_path"], args.disease)

    df_train_balanced = balance_dataset(df_train, args.disease, args.train_data_percentage, args.train_vindr_percentage)
    df_val_balanced = balance_dataset(df_val, args.disease)
    df_test_balanced = balance_dataset(df_test, args.disease)

    # Prepare datasets
    train_dataset = PneumoniaDataset(df=df_train_balanced, data_dir=data_path)
    val_dataset = PneumoniaDataset(df=df_val_balanced, data_dir=data_path)
    test_dataset = PneumoniaDataset(df=df_test_balanced, data_dir=data_path)

    # Create DataLoaders
    train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=8, shuffle=False, num_workers=4)
    test_loader = DataLoader(test_dataset, batch_size=8, shuffle=False, num_workers=4)

    # Print number of samples
    print(f"Train samples: {len(train_dataset)}")
    print(f"Validation samples: {len(val_dataset)}")
    print(f"Test samples: {len(test_dataset)}")

    # Define LoRA configuration
    lora_config = LoraConfig(
        r=args.rank,
        lora_alpha=16,
        target_modules=[
            "window_attn.fn.qkv", 
            "window_attn.fn.proj", 
            "ffn.fn.net.fc1", 
            "ffn.fn.net.fc2"
        ],
        lora_dropout=0.1,
        bias="none",
        task_type="vision"
    )


    classifier = get_medimageinsight_classifier()

    # Apply LoRA to the image encoder
    classifier.model.image_encoder = get_peft_model(classifier.model.image_encoder, lora_config)
    model = PneumoniaModel(classifier.model).to(classifier.device)

    # Collect trainable parameters
    trainable_params = [
        {'params': [p for n, p in model.base_model.named_parameters() if 'lora' in n]},
        {'params': model.classifier.parameters()}
    ]

    # Define optimizer, loss, and scheduler
    optimizer = torch.optim.AdamW(trainable_params, lr=1e-4)
    criterion = torch.nn.BCEWithLogitsLoss()
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=2, verbose=True)

        

    # Define hyperparameters
    num_epochs = 1000
    patience = 5  # Early stopping patience
    best_val_loss = float('inf')
    early_stop_counter = 0
    model_save_path = f'{args.save_path}{dataset_config["data_path"]}_{args.rank}_best_model.pth'

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

    test_probs, test_preds, test_labels = [], [], []
    with torch.no_grad():
        for images, labels in test_loader:
            image_list = [classifier.decode_base64_image(img_base64) for img_base64 in images]
            image_tensors = torch.stack([classifier.preprocess(img) for img in image_list]).to(classifier.device)
            labels = labels.float().to(classifier.device)

            logits = model(image_tensors)
            probs = torch.sigmoid(logits).squeeze().cpu().numpy()
            preds = (probs > 0.5).astype(int)

            test_probs.extend(probs)
            test_preds.extend(preds)
            test_labels.extend(labels.cpu().numpy())

    # Evaluate on Test Set using the provided evaluate_model function
    test_accuracy, test_auc, test_f1, test_cm = evaluate_model(
        probabilities=test_probs,
        predictions=test_preds,
        labels=test_labels,
        dataset_name="Test",
        bias_variables=bias_variables,
        df=df_test_balanced
    )

    wandb.finish()


if __name__ == "__main__":
    main()
