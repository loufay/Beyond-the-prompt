import os
import torch
import wandb
from config import parse_arguments, get_dataset_config
from data_loader import load_data, prepare_data
from metrics import evaluate_model
from model.model import create_knn_model
from utils import create_wandb_run_name, balance_dataset
import pandas as pd
import numpy as np


def main():
    args = parse_arguments()
    dataset_config = get_dataset_config(args.dataset)
    bias_variables = dataset_config.get("bias_variables", None)

    # Initialize W&B
    run_name = create_wandb_run_name(args, "knn")
    wandb.init(
        project=args.wandb_project,
        group=f"{args.dataset}-AdapterFT",
        name=run_name,
    )	

    output_dir = os.path.join(args.save_path, args.dataset, "knn_model")
    os.makedirs(output_dir, exist_ok=True)

    # Create balanced datasets   
    data_path = os.path.join(os.getcwd(), "MedImageInsights/data", dataset_config["data_path"])
    # Load data
    df_train, df_val, df_test = load_data(data_path, args.disease)

    # Balance data
    df_train_balanced = balance_dataset(df_train, args.disease, args.train_data_percentage, args.train_vindr_percentage)
    df_val_balanced = balance_dataset(df_val, args.disease)
    df_test_balanced = balance_dataset(df_test, args.disease)

    train_features, train_labels = prepare_data(df_train_balanced, "Train")
    val_features, val_labels = prepare_data(df_val_balanced, "Validation")
    test_features, test_labels = prepare_data(df_test_balanced, "Test")

    # Initialize model
    knn_model = create_knn_model(args.k_neighbors, args.weights)
    knn_model.fit(train_features, train_labels)

    # Evaluate the model on the validation and test sets
    val_predictions = knn_model.predict(val_features)
    val_probabilities = knn_model.predict_proba(val_features)[:, 1]
    val_accuracy, val_auc, val_f1, val_cm = evaluate_model(val_probabilities, val_predictions, val_labels, "Validation", bias_variables, df_val_balanced)
    
    test_predictions = knn_model.predict(test_features)
    test_probabilities = knn_model.predict_proba(test_features)[:, 1]
    test_accuracy, test_auc, test_f1, test_cm = evaluate_model(test_probabilities, test_predictions, test_labels, "Test", bias_variables, df_test_balanced)

    wandb.finish()


if __name__ == "__main__":
    main()
