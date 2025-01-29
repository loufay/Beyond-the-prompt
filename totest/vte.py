
from sklearn.metrics import confusion_matrix, roc_auc_score, accuracy_score, matthews_corrcoef
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
parser.add_argument("--image_processing", type=str, default="original", help="Image processing method [original, avg_all, avg_confidence]")
parser.add_argument("--text_processing", type=str, default="prompts_filtered", help="Text processing method [all, prompts_only, reports_only, prompts_imagenet, prompts_shuffled, prompts_shuffled_fixed_label, prompts_shuffled_fixed_label_small, prompts_imagenet_plus_prompt, prompts_filtered]")
args = parser.parse_args()

PATH_TO_DATA = current_dir+"/data"

args.only_no_finding = True
args.single_disease = False

bias_variables = None
run_name = create_wandb_run_name(args, experiment_type="vte")
save_path = args.save_path + args.dataset + "/" + run_name + "/"


if not os.path.exists(save_path):
    os.makedirs(save_path)
# Initialize wandb
wandb.init(
    project="MedImageInsights_6",
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
if args.only_no_finding:
    df_test_no_finding = df_test[df_test["No Finding"] == 1].sample(n=sample_size, random_state=42)
    # Load text embeddings
    if args.text_processing == "all":
        averaged_text_embeddings = np.load(f"{PATH_TO_DATA}/text_embeddings/average_embeddings_2_no_finding.npy")
    elif args.text_processing == "prompts_only":
        averaged_text_embeddings = np.load(f"{PATH_TO_DATA}/text_embeddings/average_embeddings_2_no_finding_template.npy")
    elif args.text_processing == "reports_only":
        averaged_text_embeddings = np.load(f"{PATH_TO_DATA}/text_embeddings/average_embeddings_2_no_finding_report.npy")
    elif args.text_processing == "prompts_shuffled":
        averaged_text_embeddings = np.load(f"{PATH_TO_DATA}/text_embeddings/shuffled/averaged_embeddings_shuffled_templates.npy")
    elif args.text_processing == "prompts_imagenet":
        averaged_text_embeddings = np.load(f"{PATH_TO_DATA}/text_embeddings/imagenet/averaged_embeddings_imagenet_templates.npy")
    elif args.text_processing == "prompts_shuffled_fixed_label":
        averaged_text_embeddings = np.load(f"{PATH_TO_DATA}/text_embeddings/shuffled/averaged_embeddings_shuffled_templates_fixed_label.npy")
    elif args.text_processing == "prompts_shuffled_fixed_label_small":
        averaged_text_embeddings = np.load(f"{PATH_TO_DATA}/text_embeddings/shuffled/averaged_embeddings_shuffled_templates_fixed_label_small_letter.npy")
    elif args.text_processing == "prompts_imagenet_plus_prompt":
        averaged_text_embeddings = np.load(f"{PATH_TO_DATA}/text_embeddings/imagenet/averaged_embeddings_imagenet_plus_prompts_templates.npy")
    elif args.text_processing == "prompts_filtered":
        averaged_text_embeddings = np.load(f"{PATH_TO_DATA}/text_embeddings/{args.dataset}/filtered_averaged_embeddings_{args.dataset}.npy")

    #averaged_text_embeddings = np.load(f"{PATH_TO_DATA}/text_embeddings/average_embeddings_2_no_finding.npy")
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
   # averaged_text_embeddings = np.load(f"{PATH_TO_DATA}/text_embeddings/average_embeddings_2_single_pneumonia_no_finding.npy")

print(len(df_test_no_finding))
print(len(df_test_disease))

df_test = pd.concat([df_test_no_finding, df_test_disease]).reset_index(drop=True)
# reset index
print(f"Test dataset size: {len(df_test)}")

all_predictions = []
all_true_labels = []

for idx, row in tqdm(df_test.iterrows(),total=len(df_test), desc="Zeroshot performance"):
    path_to_img =PATH_TO_DATA+row["Path"]
    img = Image.open(path_to_img).convert("L")
    # Create augmented image views and encode to base64
    if args.image_processing == "original":
        aug_image_base64_list = augment_image_to_base64(img, num_views=0)
    else:
        aug_image_base64_list = augment_image_to_base64(img, num_views=63)
    # Extract image embeddings
    augmented_image_embeddings = classifier.encode(images=aug_image_base64_list)["image_embeddings"]

    # Confidence score
    similarities = cosine_similarity(augmented_image_embeddings, averaged_text_embeddings)
    logits = torch.tensor(similarities, dtype=torch.float32)

    # Select the top 10% most confident samples
    if args.image_processing == "avg_confidence":
        top_confidence_ratio = 0.1
    else:
        top_confidence_ratio = 1
    confident_indices = select_confident_samples(logits, top_confidence_ratio)

    # Extract the embeddings of the most confident samples
    filtered_embeddings = augmented_image_embeddings[confident_indices]

    if args.image_processing != "original":
        average_filtered_embedding = filtered_embeddings.mean(axis=0)
    else:
        average_filtered_embedding = filtered_embeddings

    ## DEBUG ONLY FOR IMAGE ONLY WITHOUT AUGMENTATION
    # average_filtered_embedding = filtered_embeddings
    ##################################################
    final_scores = cosine_similarity([average_filtered_embedding], averaged_text_embeddings)  # Shape: (1, K)
    # Apply softmax for probabilities as torch tensor
    final_probabilities = torch.nn.functional.softmax(torch.tensor(final_scores), dim=1)

    # Get the predicted class
    y_true = int(row[args.disease])
    y_pred = final_probabilities.argmax(dim=1).item()

    all_predictions.append(y_pred)
    all_true_labels.append(y_true)

    # if len(all_predictions) ==10:
    #     break

# Compute metrics
conf_matrix = confusion_matrix(all_true_labels, all_predictions)
accuracy = accuracy_score(all_true_labels, all_predictions)
mcc = matthews_corrcoef(all_true_labels, all_predictions)
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
    "mcc": mcc,
})

classes = ["No "+ args.disease, args.disease]
for i, acc in enumerate(per_class_accuracy):
    print(f"Accuracy for {classes[i]}: {acc:.2f}")
    wandb.log({f"test_accuracy_{classes[i]}": acc})



# Subgroup metrics with balanced dataset
if bias_variables is not None:
    for variable, conditions in bias_variables.items():
        print(f"Evaluating bias for {variable}")

        # Exract ground truth and predictions
        y_true = [int(row[args.disease]) for idx, row in df_test.iterrows()]
        y_pred = all_predictions

        subgroup_metrics = {}
        subgroup_data = {}

        # Collect subgroup data
        for subgroup, condition in conditions.items():
            indices = df_test[condition(df_test)].index
            subgroup_y_true = [y_true[i] for i in indices if i < len(y_true)]
            subgroup_y_pred = [y_pred[i] for i in indices if i < len(y_pred)]

            # Print initial subgroup size
            print(f"  {subgroup}: {len(subgroup_y_true)} samples")

            # Store data for balancing
            subgroup_data[subgroup] = {"indices": indices, "y_true": subgroup_y_true, "y_pred": subgroup_y_pred}

  
            # Determine the smallest subgroup size
            min_size = min(len(data["y_true"]) for data in subgroup_data.values())
            print(f"Balanced size for {variable}: {min_size} samples per subgroup")

        # Balance subgroups by sampling
        for subgroup, data in subgroup_data.items():
            sampled_indices = np.random.choice(len(data["y_true"]), size=min_size, replace=False)
            subgroup_y_true = [data["y_true"][i] for i in sampled_indices]
            subgroup_y_pred = [data["y_pred"][i] for i in sampled_indices]

            # Print final balanced subgroup size
            print(f"  {subgroup}: {len(subgroup_y_true)} samples after balancing")

            # Calculate metrics
            accuracy = accuracy_score(subgroup_y_true, subgroup_y_pred)
            mcc = matthews_corrcoef(subgroup_y_true, subgroup_y_pred)

            cm_subgroup = confusion_matrix(subgroup_y_true, subgroup_y_pred)
            no_findings_accuracy = cm_subgroup[0, 0] / cm_subgroup[0].sum()
            findings_accuracy = cm_subgroup[1, 1] / cm_subgroup[1].sum()

            # Store metrics
            subgroup_metrics[subgroup] = {
                "accuracy": accuracy,
                "n_samples": min_size,
                "no_findings_accuracy": no_findings_accuracy,
                "findings_accuracy": findings_accuracy,
                "mcc": mcc,

            }
        
        # Log metrics to W&B and print
        for subgroup, metrics in subgroup_metrics.items():
            print(f"{variable} - {subgroup}: Accuracy = {metrics['accuracy']:.4f}")
            print(f"  No {args.disease}: {metrics['no_findings_accuracy']:.4f}")
            print(f"  {args.disease}: {metrics['findings_accuracy']:.4f}")
            wandb.log({
                f"{variable}_{subgroup}_accuracy": metrics["accuracy"],
                f"{variable}_{subgroup}_n_samples": metrics["n_samples"],
                f"{variable}_{subgroup}_no_findings_accuracy": metrics["no_findings_accuracy"],
                f"{variable}_{subgroup}_findings_accuracy": metrics["findings_accuracy"],
                f"{variable}_{subgroup}_mcc": metrics["mcc"],
            })
        


wandb.finish()



    



