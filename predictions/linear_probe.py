import os
import torch
import wandb
import numpy as np
import pandas as pd
from config import parse_arguments, get_dataset_config
from data_loader import load_data, prepare_samples
from metrics import evaluate_model, plot_confusion_matrix  # Reusable Evaluation Methods
from model.adapter_training import create_data_loader, create_linear_probe_model, trainer, load_trained_model, perform_inference
from utils import create_wandb_run_name, balance_dataset

def main():
    # Parse arguments and get dataset-specific configurations
    args = parse_arguments()
    dataset_config = get_dataset_config(args.dataset)
    bias_variables = dataset_config.get("bias_variables", None)

    # Initialize W&B
    run_name = create_wandb_run_name(args, "linear_probe")
    wandb.init(
        project=args.wandb_project,
        group=f"{args.dataset}-AdapterFT",
        name=run_name,
    )

    # Setup output directory
    output_dir = os.path.join(args.save_path, f"{args.dataset}-AdapterFT", f"{wandb.run.group}_{wandb.run.name}")
    os.makedirs(output_dir, exist_ok=True)

    # Load and balance datasets
    data_path = os.path.join(os.getcwd(), "MedImageInsights/data", dataset_config["data_path"])
    df_train, df_val, df_test = load_data(data_path, args.disease)

    df_train_balanced = balance_dataset(df_train, args.disease, args.train_data_percentage, args.train_vindr_percentage)
    df_val_balanced = balance_dataset(df_val, args.disease)
    df_test_balanced = balance_dataset(df_test, args.disease)

    # Prepare samples
    train_samples = prepare_samples(df_train_balanced, df_train_balanced.columns[-1024:])
    val_samples = prepare_samples(df_val_balanced, df_val_balanced.columns[-1024:])
    test_samples = prepare_samples(df_test_balanced, df_test_balanced.columns[-1024:])

    # Create DataLoaders
    train_loader = create_data_loader(train_samples, csv=df_train_balanced, mode="train", batch_size=8, num_workers=2, pin_memory=True)
    val_loader = create_data_loader(val_samples, csv=df_val_balanced, mode="val", batch_size=8, num_workers=2, pin_memory=True)
    test_loader = create_data_loader(test_samples, csv=df_test_balanced, mode="test", batch_size=8, num_workers=2, pin_memory=True)

    # Print number of samples
    print(f"Train samples: {len(train_samples['labels'])}")
    print(f"Validation samples: {len(val_samples['labels'])}")
    print(f"Test samples: {len(test_samples['labels'])}")
    
    # Define model and training parameters
    in_channels = 1024
    num_classes = 2
    learning_rate = 0.0003
    model = create_linear_probe_model(in_channels, num_classes).to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="max", factor=0.1, patience=5, verbose=True)
    loss_function = torch.nn.CrossEntropyLoss()

    # Train the model
    max_epochs = 100
    best_accuracy, best_auc = trainer(
        train_loader, 
        val_loader, 
        model, 
        loss_function, 
        optimizer, 
        scheduler, 
        max_epochs, 
        output_dir
    )


    # Perform inference on Test Set
    print("\n=== Performing Inference on Test Set ===")
    model_inference = load_trained_model(model, os.path.join(output_dir, "best_metric_model.pth"))
    predictions = perform_inference(model_inference, test_loader)

    # Extract ground truth and predicted labels
    ground_truth = [df_test_balanced[df_test_balanced["Path"] == pred["Path"]]["Pneumonia"].values[0] for pred in predictions]
    predicted_labels = [pred["PredictedClass"] for pred in predictions]
    probabilities = [pred["Probability"] for pred in predictions]


    

    # Evaluate on Test Set
    print("\n=== Evaluating on Test Set ===")
    test_accuracy, test_auc, test_f1, test_cm = evaluate_model(
        probabilities, 
        predicted_labels, 
        ground_truth, 
        "Test", 
        bias_variables, 
        df_test_balanced
    )


    wandb.finish()


if __name__ == "__main__":
    main()
