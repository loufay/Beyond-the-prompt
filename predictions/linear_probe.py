import os
import argparse
import numpy as np
import pandas as pd
import torch
import wandb
import matplotlib.pyplot as plt
import sys
from sklearn.metrics import (
    confusion_matrix,
    roc_auc_score,
    accuracy_score,
    f1_score,
    ConfusionMatrixDisplay,
)
from model.adapter_training import (
    create_data_loader,
    create_linear_probe_model,
    trainer,
    load_trained_model,
    perform_inference,
)
from utils import create_wandb_run_name, balance_dataset
current_dir = os.getcwd()
current_dir = current_dir + "/MedImageInsights"
sys.path.append(current_dir)

# Argument Parsing
parser = argparse.ArgumentParser(description="Adapter fine-tuning using Linear Probe.")
parser.add_argument("--dataset", type=str, default="MIMIC", help="Dataset to use (MIMIC, CheXpert, VinDR)")
parser.add_argument("--save_path", type=str, default=current_dir+"/Results/", help="Path to save the results")
parser.add_argument("--only_no_finding", action="store_true", help="Filter reports for 'No Finding' samples")
parser.add_argument("--single_disease", action="store_true", help="Filter reports for single disease occurrence")
parser.add_argument("--train_data_percentage", type=float, default=1.0, help="Percentage of training data to use")
parser.add_argument("--train_vindr_percentage", action="store_true", help="Percentage of training data to use")
args = parser.parse_args()

bias_variables = None

args.only_no_finding = True

# DEBUG
run_name = create_wandb_run_name(args, "linear_probe")
# Initialize W&B
wandb.init(
    project="MedImageInsights_3",
    group=f"{args.dataset}-AdapterFT",
    name=run_name,
)

# Dataset-specific configurations
PATH_TO_DATA = current_dir+"/data"
output_dir = os.path.join(args.save_path, f"{args.dataset}-AdapterFT", f"{wandb.run.group}_{wandb.run.name}")
os.makedirs(output_dir, exist_ok=True)

if args.dataset == "MIMIC":
    diseases = [
        'No Finding', 'Enlarged Cardiomediastinum', 'Cardiomegaly', 'Lung Opacity',
        'Lung Lesion', 'Edema', 'Consolidation', 'Pneumonia', 'Atelectasis',
        'Pneumothorax', 'Pleural Effusion', 'Pleural Other', 'Fracture', 'Support Devices'
    ]
    data_path = os.path.join(PATH_TO_DATA, "MIMIC-v1.0-512")
elif args.dataset == "CheXpert":
    diseases = [
        'No Finding', 'Enlarged Cardiomediastinum', 'Cardiomegaly', 'Lung Opacity',
        'Lung Lesion', 'Edema', 'Consolidation', 'Pneumonia', 'Atelectasis',
        'Pneumothorax', 'Pleural Effusion', 'Pleural Other', 'Fracture',
        'Support Devices'
    ]
    data_path = os.path.join(PATH_TO_DATA, "CheXpert-v1.0-512")
    bias_variables = {
    "sex": {"Female": lambda df: df["sex"] == "Female", "Male": lambda df: df["sex"] == "Male"},
    "age": {"Young": lambda df: df["age"] <= 62, "Old": lambda df: df["age"] > 62},
    "race": {
        "White": lambda df: df["race"] == "White",
        "Asian": lambda df: df["race"] == "Asian",
        "Black": lambda df: df["race"] == "Black",
        },
    }
elif args.dataset == "VinDR":
    diseases = [
        'No Finding', 'Bronchitis', 'Brocho-pneumonia', 'Other disease', 'Bronchiolitis',
        'Situs inversus', 'Pneumonia', 'Pleuro-pneumonia', 'Diagphramatic hernia',
        'Tuberculosis', 'Congenital emphysema', 'CPAM', 'Hyaline membrane disease',
        'Mediastinal tumor', 'Lung tumor'
    ]
    data_path = os.path.join(PATH_TO_DATA, "vindr-pcxr")
else:
    raise ValueError(f"Unsupported dataset: {args.dataset}")

# Load datasets
df_train = pd.read_csv(os.path.join(data_path, "train.csv"))
df_val = pd.read_csv(os.path.join(data_path, "val.csv"))
df_test = pd.read_csv(os.path.join(data_path, "test.csv"))

df_train = df_train[(df_train["No Finding"] == 1) | (df_train["Pneumonia"] == 1)]
df_val = df_val[(df_val["No Finding"] == 1) | (df_val["Pneumonia"] == 1)]
df_test = df_test[(df_test["No Finding"] == 1) | (df_test["Pneumonia"] == 1)]

df_train_balanced = balance_dataset(df_train, "Pneumonia", args.train_data_percentage, args.train_vindr_percentage)
df_val_balanced = balance_dataset(df_val, "Pneumonia")
df_test_balanced = balance_dataset(df_test, "Pneumonia")

print(f"Train: {len(df_train_balanced)} samples")
print(f"Val: {len(df_val_balanced)} samples")
print(f"Test: {len(df_test_balanced)} samples")


# Prepare samples
def prepare_samples(df, feature_columns):
    return {
        "img_name": df["Path"].tolist(),
        "labels": df["Pneumonia"].tolist(),
        "features": [np.array(row) for row in df[feature_columns].values]
    }

train_samples = prepare_samples(df_train_balanced, df_train_balanced.columns[-1024:])
val_samples = prepare_samples(df_val_balanced, df_val_balanced.columns[-1024:])
test_samples = prepare_samples(df_test_balanced, df_test_balanced.columns[-1024:])

# Create DataLoaders
train_loader = create_data_loader(
    train_samples,
    csv=df_train_balanced,
    mode="train",
    batch_size=8,
    num_workers=2,
    pin_memory=True,
)
val_loader = create_data_loader(
    val_samples,
    csv=df_val_balanced,
    mode="val",
    batch_size=1,
    num_workers=2,
    pin_memory=True,
)
test_loader = create_data_loader(
    test_samples,
    csv=df_test_balanced,
    mode="test",
    batch_size=1,
    num_workers=2,
    pin_memory=True,
)

# Define model and training parameters
in_channels = 1024
hidden_dim = 512
num_classes = 2
learning_rate = 0.0003

model = create_linear_probe_model(in_channels, num_classes).to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))
optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="max", factor=0.1, patience=5, verbose=True)
loss_function = torch.nn.CrossEntropyLoss()

# Train the model
max_epochs = 1000
best_accuracy, best_auc = trainer(
    train_loader, val_loader, model, loss_function, optimizer, scheduler, max_epochs, output_dir
)
print(f"Best Accuracy: {best_accuracy:.4f}")
print(f"Best AUC: {best_auc:.4f}")

# Perform inference
model_inference = load_trained_model(model, os.path.join(output_dir, "best_metric_model.pth"))
predictions = perform_inference(model_inference, test_loader)

# Extract ground truth and predicted labels
ground_truth = [df_test_balanced[df_test_balanced["Path"] == pred["Path"]]["Pneumonia"].values[0] for pred in predictions]
predicted_labels = [pred["PredictedClass"] for pred in predictions]

# Compute metrics
accuracy = accuracy_score(ground_truth, predicted_labels)
f1 = f1_score(ground_truth, predicted_labels, average="weighted")
auc = roc_auc_score(ground_truth, predicted_labels)
cm = confusion_matrix(ground_truth, predicted_labels)

no_findings_accuracy = cm[0, 0] / cm[0].sum() if cm[0].sum() > 0 else 0
pneumonia_accuracy = cm[1, 1] / cm[1].sum() if cm[1].sum() > 0 else 0

print(f"Accuracy: {accuracy:.4f}")
print(f"F1-Score: {f1:.4f}")
print(f"AUC: {auc:.4f}")
print(f"No Findings Accuracy: {no_findings_accuracy:.4f}")
print(f"Pneumonia Accuracy: {pneumonia_accuracy:.4f}")

# Plot confusion matrix
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["No Finding", "Pneumonia"])
disp.plot(cmap=plt.cm.Blues)
plt.title(f"Confusion Matrix: {wandb.run.group}_{wandb.run.name}")
plt.tight_layout()
plt.savefig(os.path.join(output_dir, "test_confusion_matrix.png"))
wandb.log({"confusion_matrix": wandb.Image(plt)})

# Log metrics to W&B
wandb.define_metric("*", step_metric=None)

# Log metrics to W&B as summary metrics
wandb.log({
    "Test_accuracy": accuracy,
    "Test_f1_score": f1,
    "Test_auc": auc,
    "Test_no_findings_accuracy": no_findings_accuracy,
    "Test_pneumonia_accuracy": pneumonia_accuracy,
    "epoch": max_epochs
}, commit=True)

# Disable automatic step tracking for all metrics
wandb.define_metric("*", step_metric=None)



# Function to calculate metrics for subgroups
def evaluate_bias(df_test, ground_truth, predicted_labels, predicted_probs, bias_variables):
    """
    Evaluate bias metrics for each subgroup defined in bias_variables.
    Args:
        df_test (DataFrame): Test DataFrame.
        ground_truth (list): Ground truth labels.
        predicted_labels (list): Predicted labels.
        predicted_probs (list): Predicted probabilities for the positive class.
        bias_variables (dict): Dictionary of bias variables and subgroups.

    Returns:
        None
    """
    for variable, subgroups in bias_variables.items():
        print(f"Evaluating bias for {variable}")

        subgroup_metrics = {}
        for subgroup, condition in subgroups.items():
            # Filter test set for the subgroup
            indices = df_test[condition(df_test)].index
            subgroup_y_true = [ground_truth[i] for i in indices if i < len(ground_truth)]
            subgroup_y_pred = [predicted_labels[i] for i in indices if i < len(predicted_labels)]
            subgroup_y_prob = [predicted_probs[i] for i in indices if i < len(predicted_probs)]

            if len(subgroup_y_true) == 0:
                continue

            # Calculate metrics
            accuracy = accuracy_score(subgroup_y_true, subgroup_y_pred)
            auc = roc_auc_score(subgroup_y_true, subgroup_y_prob) if len(np.unique(subgroup_y_true)) > 1 else float('nan')
            f1 = f1_score(subgroup_y_true, subgroup_y_pred, average="weighted")
            n_samples = len(subgroup_y_true)

            # Store metrics
            subgroup_metrics[subgroup] = {
                "accuracy": accuracy,
                "auc": auc,
                "f1_score": f1,
                "n_samples": n_samples,
            }

        # Log metrics to W&B
        for subgroup, metrics in subgroup_metrics.items():
            print(f"{variable} - {subgroup}: Accuracy = {metrics['accuracy']:.4f}, AUC = {metrics['auc']:.4f}, F1 = {metrics['f1_score']:.4f}")
            wandb.log({
                f"{variable}_{subgroup}_accuracy": metrics["accuracy"],
                f"{variable}_{subgroup}_auc": metrics["auc"],
                f"{variable}_{subgroup}_f1_score": metrics["f1_score"],
                f"{variable}_{subgroup}_n_samples": metrics["n_samples"],
                "epoch": max_epochs
            })

if bias_variables is not None:
    # Perform bias evaluation
    predicted_probs = [pred["Probability"] for pred in predictions]  # Assuming probabilities are stored in predictions
    evaluate_bias(df_test_balanced, ground_truth, predicted_labels, predicted_probs, bias_variables)

wandb.finish()
