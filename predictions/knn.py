import os
import argparse
import numpy as np
import pandas as pd
import wandb
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import (
    confusion_matrix,
    roc_auc_score,
    accuracy_score,
    f1_score,
    ConfusionMatrixDisplay,
)
import matplotlib.pyplot as plt
from utils import create_wandb_run_name, balance_dataset


# Argument Parsing
parser = argparse.ArgumentParser(description="Adapter fine-tuning using k-NN.")
parser.add_argument("--dataset", type=str, default="MIMIC", help="Dataset to use (MIMIC, CheXpert, VinDR)")
parser.add_argument("--save_path", type=str, default="/mnt/data2/datasets_lfay/MedImageInsights/Results/", help="Path to save the results")
parser.add_argument("--k_neighbors", type=int, default=5, help="Number of neighbors for k-NN")
parser.add_argument("--disease", type=str, default="Pneumonia", help="Disease to analyze")
parser.add_argument("--single_disease", action="store_true", help="Filter reports for single disease occurrence")
parser.add_argument("--only_no_finding", action="store_true", help="Filter reports for 'No Finding' samples")
parser.add_argument("--train_data_percentage", type=float, default=1.0, help="Percentage of training data to use")
parser.add_argument("--train_vindr_percentage", action="store_true", help="Percentage of training data to use")
args = parser.parse_args()

# DEBUG
# args.only_no_finding = True
# args.single_disease = True


run_name = create_wandb_run_name(args, "knn")
# Initialize W&B
wandb.init(
    project="MedImageInsights_3",
    group=f"{args.dataset}-AdapterFT",
    name=run_name,
)

# Paths and dataset-specific configurations
PATH_TO_DATA = "/mnt/data2/datasets_lfay/MedImageInsights/data"
output_dir = os.path.join(args.save_path, args.dataset, "knn_model")
os.makedirs(output_dir, exist_ok=True)
bias_variables = None

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

# Load data
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


# Prepare features and labels
def prepare_data(df, dataset_name="Train"):
    features = np.vstack(df[df.columns[-1024:]].values)
    labels = df[args.disease].values
    # print number of samples
    print(f"{dataset_name} samples: {len(labels)}")
    return features, labels

train_features, train_labels = prepare_data(df_train_balanced, "Train")
val_features, val_labels = prepare_data(df_val_balanced, "Validation")
test_features, test_labels = prepare_data(df_test_balanced, "Test")

# Initialize and train KNN model
knn_model = KNeighborsClassifier(n_neighbors=args.k_neighbors, metric="euclidean")
knn_model.fit(train_features, train_labels)


# Evaluation function
def evaluate_model(model, features, labels, dataset_name, bias_variables=None, df=None):
    """
    Evaluate the model on a dataset and log results. Optionally, evaluate subgroup metrics.

    Args:
        model: Trained model to evaluate.
        features: Feature matrix.
        labels: Ground truth labels.
        dataset_name: Name of the dataset (e.g., "Validation", "Test").
        bias_variables: Dictionary of bias variables for subgroup evaluation (optional).
    """
    # Predictions and probabilities
    predictions = model.predict(features)
    probabilities = model.predict_proba(features)[:, 1]
    
    # Metrics
    accuracy = accuracy_score(labels, predictions)
    auc = roc_auc_score(labels, probabilities)
    f1 = f1_score(labels, predictions, average="weighted")
    cm = confusion_matrix(labels, predictions)

    # Class-specific metrics
    no_findings_accuracy = cm[0, 0] / cm[0].sum() if cm[0].sum() > 0 else 0
    pneumonia_accuracy = cm[1, 1] / cm[1].sum() if cm[1].sum() > 0 else 0

    print(f"{dataset_name} Accuracy: {accuracy:.4f}")
    print(f"{dataset_name} AUC: {auc:.4f}")
    print(f"{dataset_name} F1-Score: {f1:.4f}")

    # Log main metrics
    wandb.log({
        f"{dataset_name}_accuracy": accuracy,
        f"{dataset_name}_auc": auc,
        f"{dataset_name}_f1_score": f1,
        f"{dataset_name}_no_findings_accuracy": no_findings_accuracy,
        f"{dataset_name}_pneumonia_accuracy": pneumonia_accuracy,
    })

    # Plot confusion matrix
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["No Finding", args.disease])
    disp.plot(cmap=plt.cm.Blues)
    plt.title(f"{dataset_name} Confusion Matrix")
    plt.savefig(os.path.join(output_dir, f"{dataset_name}_confusion_matrix.png"))
    wandb.log({f"{dataset_name}_confusion_matrix": wandb.Image(plt)})
    plt.close()

    # Subgroup metrics (if bias variables are provided)
    if (bias_variables and df is not None) and (dataset_name == "Test"):
        print(f"\nEvaluating Subgroup Metrics for {dataset_name}...")
        y_true = labels
        y_prob = probabilities
        y_pred = predictions

        for variable, conditions in bias_variables.items():
            print(f"\nAnalyzing bias for {variable}...")
            subgroup_metrics = {}
            subgroup_data = {}

            # Collect subgroup data
            print(f"Initial subgroup sizes for {variable}:")
            for subgroup, condition in conditions.items():
                indices = df[condition(df)].index
                subgroup_y_true = [y_true[i] for i in indices if i < len(y_true)]
                subgroup_y_pred = [y_pred[i] for i in indices if i < len(y_pred)]
                subgroup_y_prob = [y_prob[i] for i in indices if i < len(y_prob)]

                print(f"  {subgroup}: {len(subgroup_y_true)} samples")

                subgroup_data[subgroup] = {
                    "indices": indices,
                    "y_true": subgroup_y_true,
                    "y_pred": subgroup_y_pred,
                    "y_prob": subgroup_y_prob,
                }

            # Balance subgroups
            min_size = min(len(data["y_true"]) for data in subgroup_data.values())
            print(f"Balanced size for {variable}: {min_size} samples per subgroup")

            for subgroup, data in subgroup_data.items():
                sampled_indices = np.random.choice(len(data["y_true"]), size=min_size, replace=False)
                subgroup_y_true = [data["y_true"][i] for i in sampled_indices]
                subgroup_y_pred = [data["y_pred"][i] for i in sampled_indices]
                subgroup_y_prob = [data["y_prob"][i] for i in sampled_indices]

                print(f"  {subgroup}: {len(subgroup_y_true)} samples after balancing")

                # Calculate subgroup metrics
                accuracy = accuracy_score(subgroup_y_true, subgroup_y_pred)
                roc_auc = (
                    roc_auc_score(subgroup_y_true, subgroup_y_prob)
                    if len(np.unique(subgroup_y_true)) > 1
                    else float('nan')
                )

                # Store subgroup metrics
                subgroup_metrics[subgroup] = {
                    "accuracy": accuracy,
                    "roc_auc": roc_auc,
                    "n_samples": min_size,
                }

            # Log subgroup metrics to W&B
            for subgroup, metrics in subgroup_metrics.items():
                print(f"{variable} - {subgroup}: Accuracy = {metrics['accuracy']:.4f}, AUC = {metrics['roc_auc']:.4f}")
                wandb.log({
                    f"{dataset_name}_{variable}_{subgroup}_accuracy": metrics["accuracy"],
                    f"{dataset_name}_{variable}_{subgroup}_roc_auc": metrics["roc_auc"],
                    f"{dataset_name}_{variable}_{subgroup}_n_samples": metrics["n_samples"],
                })

    return accuracy, auc, f1, cm


# Evaluate the model on the validation and test sets
val_accuracy, val_auc, val_f1, val_cm = evaluate_model(knn_model, val_features, val_labels, "Validation", bias_variables, df_val_balanced)
test_accuracy, test_auc, test_f1, test_cm = evaluate_model(knn_model, test_features, test_labels, "Test", bias_variables, df_test_balanced)

wandb.finish()