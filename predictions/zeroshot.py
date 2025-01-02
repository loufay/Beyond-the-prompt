import os
import sys
import base64
import argparse
import pandas as pd
import numpy as np
import torch
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, roc_auc_score, accuracy_score, matthews_corrcoef
from tqdm import tqdm
import wandb
from utils import read_image, create_wandb_run_name, calculate_subgroup_metrics, balance_dataset
current_dir = os.getcwd()
current_dir = current_dir + "/MedImageInsights"
sys.path.append(current_dir)

from MedImageInsight.medimageinsightmodel import MedImageInsight

# Argument Parsing
parser = argparse.ArgumentParser(description="Extract findings and impressions from radiology reports.")
parser.add_argument("--dataset", type=str, default="CheXpert", help="Dataset to use (MIMIC, CheXpert, VinDR)")
parser.add_argument("--save_path", type=str, default=current_dir+"/Results/", help="Path to save the results")
parser.add_argument("--disease", type=str, default="Pneumonia", help="Disease to analyze")
parser.add_argument("--single_disease", action="store_true", help="Filter reports for single disease occurrence")
parser.add_argument("--only_no_finding", action="store_true", help="Filter reports for 'No Finding' samples")
args = parser.parse_args()

PATH_TO_DATA = os.path.join(current_dir, "data")

# Generate unique run name
base_run_name = create_wandb_run_name(args, "zeroshot")
bias_variables = None

# Load Dataset
if args.dataset == "MIMIC":
    diseases = [
        'No Finding', 'Enlarged Cardiomediastinum', 'Cardiomegaly', 'Lung Opacity',
        'Lung Lesion', 'Edema', 'Consolidation', 'Pneumonia', 'Atelectasis',
        'Pneumothorax', 'Pleural Effusion', 'Pleural Other', 'Fracture', 'Support Devices'
    ]
    read_path = os.path.join(PATH_TO_DATA, "MIMIC-v1.0-512")
elif args.dataset == "CheXpert":
    diseases = ['No Finding', 'Enlarged Cardiomediastinum', 'Cardiomegaly', 'Lung Opacity',
            'Lung Lesion', 'Edema', 'Consolidation', 'Pneumonia', 'Atelectasis',
            'Pneumothorax', 'Pleural Effusion', 'Pleural Other', 'Fracture',
            'Support Devices']
    read_path = os.path.join(PATH_TO_DATA, "CheXpert-v1.0-512")

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
    diseases = ['No Finding', 'Bronchitis', 'Brocho-pneumonia', 'Other disease', 'Bronchiolitis', 'Situs inversus', 'Pneumonia', 'Pleuro-pneumonia', 'Diagphramatic hernia', 'Tuberculosis', 'Congenital emphysema', 'CPAM', 'Hyaline membrane disease', 'Mediastinal tumor', 'Lung tumor']
    read_path = os.path.join(PATH_TO_DATA, "vindr-pcxr")


df_train = pd.read_csv(os.path.join(read_path, "train.csv"))
df_test = pd.read_csv(os.path.join(read_path, "test.csv"))

# Filter test data for single disease (other diseases not present) or disease + other diseases
if args.single_disease:
    # e.g. only Pneumonia is true
    single_disease = diseases.copy()
    single_disease.remove(args.disease)
    finding_samples_test = df_test[
        (df_test[args.disease] == 1) & (df_test[single_disease] == 0).all(axis=1)
    ]
else:
    # e.g. Pneumonia is true and other diseases can be true
    finding_samples_test = df_test[df_test[args.disease] == 1]

# Filter test data for 'No Finding' samples or anything can be true as long as args.disease is false
if args.only_no_finding:
    df_train = df_train[(df_train["No Finding"] == 1) | (df_train["Pneumonia"] == 1)]
    df_test = df_test[(df_test["No Finding"] == 1) | (df_test["Pneumonia"] == 1)]

    no_disease = "No Finding"
else:
    no_finding_samples_test = df_test[df_test[args.disease] == 0].sample(len(finding_samples_test), random_state=42)
    no_disease = "No "+args.disease

df_test_balanced = balance_dataset(df_test, "Pneumonia")

#filtered_test_images = pd.concat([no_finding_samples_test, finding_samples_test], ignore_index=True)

# Initialize Model
classifier = MedImageInsight(
    model_dir=current_dir+ "/MedImageInsight/2024.09.27",
    vision_model_name="medimageinsigt-v1.0.0.pt",
    language_model_name="language_model.pth"
)
classifier.load_model()
classifier.model.eval()

# Encode Images
images = [
    base64.encodebytes(read_image(PATH_TO_DATA + row["Path"])).decode("utf-8")
    for _, row in df_test_balanced.iterrows()
]

# Perform Zero-Shot Predictions
add_ons = ["x-ray chest anteroposterior ", "x-ray chest ", "x-ray ", "chest ", ""]
chunk_size = 10
num_images = len(images)

for add_on in add_ons:
    # Start a new W&B run for each add_on
    run_name = f"{base_run_name}_{add_on.strip().replace(' ', '_')}"
    wandb.init(
        project="MedImageInsights_5",
        group=f"{args.dataset}-ZeroShot",
        name=run_name,
        reinit=True
    )
    save_path = os.path.join(args.save_path, args.dataset, run_name)
    os.makedirs(save_path, exist_ok=True)

    print(f"Processing >{add_on}< dataset")
    prompt_diseases = [f"{add_on}{class_name}" for class_name in [no_disease, args.disease]]

    ground_truth = []
    all_predictions = []
    for _, row in df_test_balanced.iterrows():
        #labels_dict = {disease: row[disease[len(add_on):]] for disease in prompt_diseases}
        labels_dict = {
            prompt_diseases[0]: int(row[args.disease] == 0),
            prompt_diseases[1]: int(row[args.disease] == 1)
        }
        ground_truth.append(labels_dict)

    # Print dataset statistics
    # print(f"Number of {no_disease} Images: {len(no_finding_samples_test)}")
    # print(f"Number of {args.disease} Images: {len(finding_samples_test)}")
    print(f"Number of {no_disease} Images: {len(df_test_balanced[df_test_balanced[args.disease] == 0])}")
    print(f"Number of {args.disease} Images: {len(df_test_balanced[df_test_balanced[args.disease] == 1])}")

    for start_idx in tqdm(range(0, num_images, chunk_size), desc="Zero-shot prediction"):
        end_idx = min(start_idx + chunk_size, num_images)
        image_chunk = images[start_idx:end_idx]
        label_chunk = ground_truth[start_idx:end_idx]

        with torch.no_grad():
            predictions = classifier.predict(image_chunk, prompt_diseases, multilabel=False)
        all_predictions.extend(predictions)


    # Compute Metrics
    y_true = [
        next((i for i, disease in enumerate(prompt_diseases) if truth[disease] == 1), None)
        for truth in ground_truth
    ]
    y_true = y_true[:len(all_predictions)]
    y_pred = [
        max(range(len(prompt_diseases)), key=lambda i: pred[prompt_diseases[i]])
        for pred in all_predictions
    ]

    # Overall Metrics
    accuracy = accuracy_score(y_true, y_pred)
    conf_matrix = confusion_matrix(y_true, y_pred)
    if len(np.unique(y_true)) > 1:
        roc_auc = roc_auc_score(
            [[truth[disease] for disease in prompt_diseases] for truth in ground_truth],
            [[pred[disease] for disease in prompt_diseases] for pred in all_predictions]
        )
    else:
        roc_auc = float('nan')
 
        # Per-Class Accuracy
    per_class_accuracy = conf_matrix.diagonal() / conf_matrix.sum(axis=1)
    mcc = matthews_corrcoef(y_true, y_pred)

    print(f"Overall Accuracy: {accuracy}")
    print(f"Confusion Matrix:\n{conf_matrix}")
    print(f"ROC AUC: {roc_auc}")
    print(f"MCC: {mcc}")

    # Log metrics to W&B
    plt.figure(figsize=(8, 6))
    stanford_palette = ["#F6C2C2", "#ED8686", "#E44949", "#8C1515", "#691010", "#460B0B"]
    sns.heatmap(conf_matrix, annot=True, fmt='d', cmap=sns.blend_palette(stanford_palette, as_cmap=True), cbar=False,
                xticklabels=[no_disease,args.disease], yticklabels=[no_disease,args.disease], annot_kws={"fontsize": 32})

    plt.title("Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.tight_layout()
    plt.savefig(f"{save_path}confusion_matrix_test.png")

    wandb.log({
        "test_accuracy": accuracy,
        "test_roc_auc": roc_auc,
        "confusion_matrix": wandb.Image(f"{save_path}confusion_matrix_test.png"),
        "mcc": mcc
    })

    for i, acc in enumerate(per_class_accuracy):
        if add_on == "":
            class_name = prompt_diseases[i]
        else:
            class_name = prompt_diseases[i].split(add_on)[-1]
        print(f"Accuracy for {class_name}: {acc:.2f}")
        wandb.log({f"test_accuracy_{class_name}": acc})
    
    # Subgroup Metrics
    # Subgroup Metrics with Balanced Sizes
    if bias_variables is not None:
        for variable, conditions in bias_variables.items():
            print(f"Evaluating bias for {variable}")

            # Extract ground truth and predictions
            y_true = [int(row[args.disease]) for _, row in df_test_balanced.iterrows()]
            y_prob = [pred[prompt_diseases[1]] for pred in all_predictions]  # Probability for 'Pneumonia'
            y_pred = [1 if prob > 0.5 else 0 for prob in y_prob]  # Binary predictions

            subgroup_metrics = {}
            subgroup_data = {}

            # Collect subgroup data
            print(f"Initial subgroup sizes for {variable}:")
            for subgroup, condition in conditions.items():
                indices = df_test[condition(df_test)].index
                subgroup_y_true = [y_true[i] for i in indices if i < len(y_true)]
                subgroup_y_pred = [y_pred[i] for i in indices if i < len(y_pred)]
                subgroup_y_prob = [y_prob[i] for i in indices if i < len(y_prob)]

                # Print initial subgroup size
                print(f"  {subgroup}: {len(subgroup_y_true)} samples")

                # Store data for balancing
                subgroup_data[subgroup] = {
                    "indices": indices,
                    "y_true": subgroup_y_true,
                    "y_pred": subgroup_y_pred,
                    "y_prob": subgroup_y_prob,
                }

            # Determine the smallest subgroup size
            min_size = min(len(data["y_true"]) for data in subgroup_data.values())
            print(f"Balanced size for {variable}: {min_size} samples per subgroup")

            # Balance subgroups by sampling
            for subgroup, data in subgroup_data.items():
                sampled_indices = np.random.choice(len(data["y_true"]), size=min_size, replace=False)
                subgroup_y_true = [data["y_true"][i] for i in sampled_indices]
                subgroup_y_pred = [data["y_pred"][i] for i in sampled_indices]
                subgroup_y_prob = [data["y_prob"][i] for i in sampled_indices]

                # Print final balanced subgroup size
                print(f"  {subgroup}: {len(subgroup_y_true)} samples after balancing")

                # Calculate metrics
                accuracy = accuracy_score(subgroup_y_true, subgroup_y_pred)
                roc_auc = roc_auc_score(
                    subgroup_y_true,  # Ground truth labels (integers)
                    subgroup_y_prob   # Predicted probabilities for the positive class
                ) if len(np.unique(subgroup_y_true)) > 1 else float('nan')

                # confusion matrix
                cm_subgroup = confusion_matrix(subgroup_y_true, subgroup_y_pred)
                no_findings_accuracy = cm_subgroup[0, 0] / cm_subgroup[0].sum()
                findings_accuracy = cm_subgroup[1, 1] / cm_subgroup[1].sum()
                mcc_subgroup = matthews_corrcoef(subgroup_y_true, subgroup_y_pred)

                # Store metrics
                subgroup_metrics[subgroup] = {
                    "accuracy": accuracy,
                    "roc_auc": roc_auc,
                    "n_samples": min_size,
                    "no_findings_accuracy": no_findings_accuracy,
                    "findings_accuracy": findings_accuracy,
                    "mcc": mcc_subgroup
                }

            # Log metrics to W&B and print
            for subgroup, metrics in subgroup_metrics.items():
                print(f"{variable} - {subgroup}: Accuracy = {metrics['accuracy']:.4f}, AUC = {metrics['roc_auc']:.4f}")
                wandb.log({
                    f"{variable}_{subgroup}_accuracy": metrics["accuracy"],
                    f"{variable}_{subgroup}_roc_auc": metrics["roc_auc"],
                    f"{variable}_{subgroup}_n_samples": metrics["n_samples"],
                    f"{variable}_{subgroup}_no_findings_accuracy": metrics["no_findings_accuracy"],
                    f"{variable}_{subgroup}_findings_accuracy": metrics["findings_accuracy"],
                    f"{variable}_{subgroup}_mcc": metrics["mcc"]
                })


    wandb.finish()

