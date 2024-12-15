

from sklearn.metrics import confusion_matrix, roc_auc_score, accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
import base64
import os
import torch
from sklearn.metrics.pairwise import cosine_similarity
from utils import read_image, zero_shot_prediction, extract_findings_and_impressions, create_wandb_run_name
import sys
import wandb
from tqdm import tqdm
current_dir = os.getcwd()
current_dir = current_dir + "/MedImageInsights"
sys.path.append(current_dir)
from MedImageInsight.medimageinsightmodel import MedImageInsight
import argparse

# Read arguments
parser = argparse.ArgumentParser(description="Extract findings and impressions from radiology reports.")
parser.add_argument("--dataset", type=str, default="VinDR", help="Dataset to use (MIMIC, CheXpert, VinDR)")
parser.add_argument("--compare_to_mimic", action="store_true", help="Compare to MIMIC reports")
parser.add_argument("--findings_only", action="store_true", help="Extract only the findings section")
parser.add_argument("--impression_only", action="store_true", help="Extract only the impression section")
parser.add_argument("--combined", action="store_true", help="Combine findings and impressions")
parser.add_argument("--save_path", type=str, default=current_dir+"/Results/", help="Path to save the results")
parser.add_argument("--disease", type=str, default="Pneumonia", help="Disease to analyze")
parser.add_argument("--single_disease", action="store_true", help="Filter reports for single disease occurrence")
parser.add_argument("--only_no_finding", action="store_true", help="Filter reports for 'No Finding' samples")
parser.add_argument("--nr_reports_per_disease", type=int, default=10, help="Number of reports to sample per disease")
args = parser.parse_args()

PATH_TO_DATA = os.path.join(current_dir, "data")

##DEBUG
# args.findings_only = True
# args.only_no_finding = True
# args.compare_to_mimic = True

# Set options
findings_only = args.findings_only
impression_only = args.impression_only
combined = args.combined


run_name = create_wandb_run_name(args, experiment_type="report")

save_path = args.save_path + args.dataset + "/" + run_name + "/"
if not os.path.exists(save_path):
    os.makedirs(save_path)
# Initialize wandb
wandb.init(
    project="MedImageInsights_3",
    group=f"{args.dataset}-Report-ZeroShot",
    name=run_name,
)

# Load training and test datasets
if args.dataset == "MIMIC":
    read_path = PATH_TO_DATA+"/MIMIC-v1.0-512/"

    diseases = ['No Finding', 'Enlarged Cardiomediastinum', 'Cardiomegaly', 'Lung Opacity',
            'Lung Lesion', 'Edema', 'Consolidation', 'Pneumonia', 'Atelectasis',
            'Pneumothorax', 'Pleural Effusion', 'Pleural Other', 'Fracture',
            'Support Devices']
    
elif args.dataset == "CheXpert":
    diseases = ['No Finding', 'Enlarged Cardiomediastinum', 'Cardiomegaly', 'Lung Opacity',
            'Lung Lesion', 'Edema', 'Consolidation', 'Pneumonia', 'Atelectasis',
            'Pneumothorax', 'Pleural Effusion', 'Pleural Other', 'Fracture',
            'Support Devices']
    read_path = PATH_TO_DATA+"/CheXpert-v1.0-512/"

elif args.dataset == "VinDR":
    diseases = ['No Finding', 'Bronchitis', 'Brocho-pneumonia', 'Other disease', 'Bronchiolitis', 'Situs inversus', 'Pneumonia', 'Pleuro-pneumonia', 'Diagphramatic hernia', 'Tuberculosis', 'Congenital emphysema', 'CPAM', 'Hyaline membrane disease', 'Mediastinal tumor', 'Lung tumor']
    read_path = PATH_TO_DATA+"/vindr-pcxr/"

    diseases_mimic = ['No Finding', 'Enlarged Cardiomediastinum', 'Cardiomegaly', 'Lung Opacity',
            'Lung Lesion', 'Edema', 'Consolidation', 'Pneumonia', 'Atelectasis',
            'Pneumothorax', 'Pleural Effusion', 'Pleural Other', 'Fracture',
            'Support Devices']

if args.compare_to_mimic:
    read_path_mimic = PATH_TO_DATA+"/MIMIC-v1.0-512/"
    df_train = pd.read_csv(read_path_mimic + "train.csv")
else:
    df_train = pd.read_csv(read_path + "train.csv")

df_test = pd.read_csv(read_path + "test.csv")

# drop finding from diseases to ensure only one disease is present 
if args.single_disease:
    single_disease = diseases.copy()
    single_disease.remove(args.disease)
    finding_samples_train = df_train[(df_train[args.disease] == 1) & (df_train[single_disease]== 0).all(axis=1)]
else:
    finding_samples_train = df_train[df_train[args.disease] == 1]

# Filter test data for 'No Finding' samples or anything can be true as long as args.disease is false
if args.only_no_finding:
    if args.compare_to_mimic:
        no_finding_samples_train = df_train[(df_train['No Finding'] == 1) & (df_train[diseases_mimic[1:]]== 0).all(axis=1)]
    else:
        no_finding_samples_train = df_train[(df_train['No Finding'] == 1) & (df_train[diseases[1:]]== 0).all(axis=1)]
    no_finding_samples_train = no_finding_samples_train.sample(len(finding_samples_train), random_state=42)
    no_disease = "No Finding"
else:
    no_finding_samples_train = df_train[df_train[args.disease] == 0]
    no_finding_samples_train = no_finding_samples_train.sample(len(finding_samples_train), random_state=42)
    no_disease = "No "+args.disease

## 1. Extract reports to compare to
if not args.findings_only and not args.impression_only:
    no_finding_reports = no_finding_samples_train.report.sample(args.nr_reports_per_disease, random_state=42)
    finding_reports = finding_samples_train.report.sample(args.nr_reports_per_disease, random_state=42)
elif args.findings_only:
    print("Extracting findings only")
    no_finding_reports = no_finding_samples_train.section_findings.sample(args.nr_reports_per_disease*2, random_state=42)
    finding_reports = finding_samples_train.section_findings.sample(args.nr_reports_per_disease*2, random_state=42)
    # drop if nan
    no_finding_reports = no_finding_reports.dropna()[0:args.nr_reports_per_disease]
    finding_reports = finding_reports.dropna()[0:args.nr_reports_per_disease]

elif args.impression_only:
    print("Extracting impressions only")
    no_finding_reports = no_finding_samples_train.section_findings.sample(args.nr_reports_per_disease*2, random_state=42)
    finding_reports = finding_samples_train.section_findings.sample(args.nr_reports_per_disease*2, random_state=42)
    # drop if nan
    no_finding_reports = no_finding_reports.dropna()[0:args.nr_reports_per_disease]
    finding_reports = finding_reports.dropna()[0:args.nr_reports_per_disease]

elif args.combined:
    print("Extracting combined findings and impressions")
    # Combine findings and impressions for no-finding and finding samples
    no_finding_reports = (
        no_finding_samples_train.section_findings.str.cat(
            no_finding_samples_train.section_impression, sep=" ", na_rep=""
        )
        .sample(args.nr_reports_per_disease * 2, random_state=42)
    )
    finding_reports = (
        finding_samples_train.section_findings.str.cat(
            finding_samples_train.section_impression, sep=" ", na_rep=""
        )
        .sample(args.nr_reports_per_disease * 2, random_state=42)
    )

    # drop nan
    no_finding_reports = no_finding_reports.dropna()[0:args.nr_reports_per_disease]
    finding_reports = finding_reports.dropna()[0:args.nr_reports_per_disease]


print(f"Number of {no_disease} Reports: {len(no_finding_reports)}")
print(f"Number of {args.disease} Reports: {len(finding_reports)}")


## 2. Extract images from TEST data
if args.single_disease:
    finding_samples_test = df_test[(df_test[args.disease] == 1) & (df_test[single_disease]== 0).all(axis=1)]
else:
    finding_samples_test = df_test[df_test[args.disease] == 1]

if args.only_no_finding:
    no_finding_samples_test = df_test[(df_test['No Finding'] == 1) & (df_test[diseases[1:]]== 0).all(axis=1)]
else:
    no_finding_samples_test = df_test[df_test[args.disease] == 0]

no_finding_samples_test = no_finding_samples_test.sample(len(finding_samples_test), random_state=42)
filtered_test_images = pd.concat([no_finding_samples_test, finding_samples_test], ignore_index=True)
print(f"Number of {no_disease} Images: {len(no_finding_samples_test)}")
print(f"Number of {args.disease} Images: {len(finding_samples_test)}")

## 3. Initialize model
classifier = MedImageInsight(
    model_dir=current_dir+"/MedImageInsight/2024.09.27",
    vision_model_name="medimageinsigt-v1.0.0.pt",
    language_model_name="language_model.pth"
)
classifier.load_model()

# Encode the selected reports
with torch.no_grad():
    report_texts = list(no_finding_reports) + list(finding_reports)
    report_embeddings = classifier.encode(texts=report_texts)["text_embeddings"]

# Create labels for the reports
report_labels = [0] * len(no_finding_reports) + [1] * len(finding_reports)

## 4. Zero-shot prediction

all_predictions = []
all_labels = []

for i, row in tqdm(filtered_test_images.iterrows(),total=len(filtered_test_images), desc="Zeroshot performance"):
    path_to_img =PATH_TO_DATA+row["Path"]
    image_base64 = base64.encodebytes(read_image(path_to_img)).decode("utf-8")
    
    # Encode the image
    image_embedding = classifier.encode(images=[image_base64])["image_embeddings"]

    # Predict based on the closest reports
    predicted_label, closest_indices = zero_shot_prediction(image_embedding, report_embeddings, report_labels, k=5)

    # Get ground truth label for evaluation
    true_label = 0 if row[args.disease] == 0 else 1
    all_predictions.append(predicted_label)
    all_labels.append(true_label)


## 5. Evaluate the predictions
accuracy = accuracy_score(all_labels, all_predictions)
conf_matrix = confusion_matrix(all_labels, all_predictions)
roc_auc = roc_auc_score(all_labels, all_predictions)

# Compute per-class accuracy
per_class_accuracy = conf_matrix.diagonal() / conf_matrix.sum(axis=1)

print(f"Overall Accuracy: {accuracy}")
print(f"Confusion Matrix:\n{conf_matrix}")
print(f"ROC AUC: {roc_auc}")
for i, acc in enumerate(per_class_accuracy):
    class_name = no_disease if i == 0 else args.disease
    print(f"Accuracy for {class_name}: {acc:.2f}")

# Plot confusion matrix
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
})

# log per class accuracy
for i, acc in enumerate(per_class_accuracy):
    class_name = no_disease if i == 0 else args.disease
    wandb.log({f"test_accuracy_{class_name}": acc})


















