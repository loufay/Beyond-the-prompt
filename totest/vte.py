
from sklearn.metrics import confusion_matrix, roc_auc_score, accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
import base64
import os
import sys
import torch
from sklearn.metrics.pairwise import cosine_similarity
import sys
import wandb
from tqdm import tqdm
current_dir = os.getcwd()
current_dir = current_dir + "/MedImageInsights"
sys.path.append(current_dir)
from MedImageInsight.medimageinsightmodel import MedImageInsight
current_dir_pred = current_dir + "/predictions"
sys.path.append(current_dir_pred)
from utils import read_image, zero_shot_prediction, extract_findings_and_impressions, create_wandb_run_name, chest_xray_augmentations, augment_image_to_base64, select_confident_samples
import argparse
from PIL import Image, ImageOps

# Read arguments
parser = argparse.ArgumentParser(description="Extract findings and impressions from radiology reports.")
parser.add_argument("--dataset", type=str, default="MIMIC", help="Dataset to use (MIMIC, CheXpert, VinDR)")
parser.add_argument("--save_path", type=str, default=current_dir+"/Results/", help="Path to save the results")
parser.add_argument("--disease", type=str, default="Pneumonia", help="Disease to analyze")
parser.add_argument("--single_disease", action="store_true", help="Filter reports for single disease occurrence")
parser.add_argument("--only_no_finding", action="store_true", help="Filter reports for 'No Finding' samples")
parser.add_argument("--nr_reports_per_disease", type=int, default=10, help="Number of reports to sample per disease")
parser.add_argument("--image_processing", type=str, default="avg_confidence", help="Image processing method [original, avg_all, avg_confidence]")
parser.add_argument("--text_processing", type=str, default="all", help="Text processing method [all, prompts_only, reports_only]")
args = parser.parse_args()

PATH_TO_DATA = current_dir+"/data"

args.only_no_finding = True
args.single_disease = False


run_name = create_wandb_run_name(args, experiment_type="vte")
save_path = args.save_path + args.dataset + "/" + run_name + "/"


if not os.path.exists(save_path):
    os.makedirs(save_path)
# Initialize wandb
wandb.init(
    project="MedImageInsights_4",
    group=f"{args.dataset}-VTE",
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

## Initialize model
classifier = MedImageInsight(
    model_dir=current_dir+"/MedImageInsight/2024.09.27",
    vision_model_name="medimageinsigt-v1.0.0.pt",
    language_model_name="language_model.pth"
)
classifier.load_model()

##  EMBEDDINGS
# Load images
df_test = pd.read_csv(read_path + "test.csv")
# shuffle df_test
df_test = df_test.sample(frac=1, random_state=42).reset_index(drop=True)

# balance dataset to have equal number of positive and negative samples 
sample_size = min(df_test[args.disease].value_counts().values)
# sample_size = 10
if args.only_no_finding:
    df_test_no_finding = df_test[df_test["No Finding"] == 1].sample(n=sample_size, random_state=42)
    # Load text embeddings
    averaged_text_embeddings = np.load(f"{PATH_TO_DATA}/text_embeddings/average_embeddings_2_no_finding.npy")
else:
    df_test_no_finding = df_test[df_test[args.disease] == 0].sample(n=sample_size, random_state=42)
    # Load text embeddings
    averaged_text_embeddings = np.load(f"{PATH_TO_DATA}/text_embeddings/average_embeddings_2.npy")

if args.single_disease:
    single_disease = diseases.copy()
    single_disease.remove(args.disease)
    df_test_disease = df_test[(df_test[args.disease] == 1) & (df_test[single_disease] == 0).all(axis=1)]
    df_test_disease = df_test_disease.sample(n=sample_size, random_state=42)
    averaged_text_embeddings = np.load(f"{PATH_TO_DATA}/text_embeddings/average_embeddings_2_single_pneumonia_no_finding.npy")

else:
    df_test_disease = df_test[df_test[args.disease] == 1].sample(n=sample_size, random_state=42)
    averaged_text_embeddings = np.load(f"{PATH_TO_DATA}/text_embeddings/average_embeddings_2_single_pneumonia_no_finding.npy")

print(len(df_test_no_finding))
print(len(df_test_disease))

df_test = pd.concat([df_test_no_finding, df_test_disease])
print(f"Test dataset size: {len(df_test)}")

all_predictions = []
all_true_labels = []

for idx, row in tqdm(df_test.iterrows(),total=len(df_test), desc="Zeroshot performance"):
    path_to_img =PATH_TO_DATA+row["Path"]
    img = Image.open(path_to_img).convert("L")
    # Create augmented image views and encode to base64
    aug_image_base64_list = augment_image_to_base64(img, num_views=63)
    # Extract image embeddings
    augmented_image_embeddings = classifier.encode(images=aug_image_base64_list)["image_embeddings"]

    # Confidence score
    similarities = cosine_similarity(augmented_image_embeddings, averaged_text_embeddings)
    logits = torch.tensor(similarities, dtype=torch.float32)

    # Select the top 10% most confident samples
    top_confidence_ratio = 0.1  # 10%
    #top_confidence_ratio = 1
    confident_indices = select_confident_samples(logits, top_confidence_ratio)

    # Extract the embeddings of the most confident samples
    filtered_embeddings = augmented_image_embeddings[confident_indices]
    average_filtered_embedding = filtered_embeddings.mean(axis=0)

    ## DEBUG ONLY FOR IMAGE ONLY WITHOUT AUGMENTATION
    # average_filtered_embedding = filtered_embeddings
    ##################################################
    final_scores = cosine_similarity([average_filtered_embedding], averaged_text_embeddings)  # Shape: (1, K)
    # Apply softmax for probabilities as torch tensor
    final_probabilities = torch.nn.functional.softmax(torch.tensor(final_scores), dim=1)

    # final_scores = cosine_similarity(filtered_embeddings, averaged_text_embeddings)  # Shape: (6, K)
    # Apply softmax for probabilities as torch tensor
    # final_probabilities = torch.nn.functional.softmax(torch.tensor(final_scores), dim=1)
    # get for each image the max probability
   # final_probabilities = final_probabilities.max(dim=1).values
    # decide for the class which is predicted most often for the images
   # y_preds = final_probabilities.argmax(dim=1)
    # y_pred = 1 if y_preds.sum() > int(len(y_preds)/2) else 0

    # Get the predicted class
    y_true = int(row[args.disease])
    y_pred = final_probabilities.argmax(dim=1).item()

    all_predictions.append(y_pred)
    all_true_labels.append(y_true)

    if len(all_predictions) ==100:
        break

# Compute metrics
conf_matrix = confusion_matrix(all_true_labels, all_predictions)
accuracy = accuracy_score(all_true_labels, all_predictions)
#roc_auc = roc_auc_score(all_true_labels, all_predictions)

# Log metrics to W&B
plt.figure(figsize=(8, 6))
stanford_palette = ["#F6C2C2", "#ED8686", "#E44949", "#8C1515", "#691010", "#460B0B"]
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap=sns.blend_palette(stanford_palette, as_cmap=True), cbar=False,
            xticklabels=["No "+ args.disease ,args.disease], yticklabels=["No "+ args.disease,args.disease], annot_kws={"fontsize": 32})

plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("True")
plt.tight_layout()
plt.savefig(f"{save_path}confusion_matrix_test.png")

# Per class accuracy
per_class_accuracy = conf_matrix.diagonal() / conf_matrix.sum(axis=1)

print(f"Overall Accuracy: {accuracy}")
print(f"Confusion Matrix:\n{conf_matrix}")

# Save results
wandb.log({
    "test_accuracy": accuracy,
    "confusion_matrix": wandb.Image(f"{save_path}confusion_matrix_test.png"),
})

classes = ["No "+ args.disease, args.disease]
for i, acc in enumerate(per_class_accuracy):
    print(f"Accuracy for {classes[i]}: {acc:.2f}")
    wandb.log({f"test_accuracy_{classes[i]}": acc})

wandb.finish()



    



