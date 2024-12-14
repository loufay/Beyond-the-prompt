from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import re
from sklearn.metrics import confusion_matrix, roc_auc_score, accuracy_score
from PIL import Image, ImageOps
from torchvision import transforms
from io import BytesIO
import base64
import random
import torch
from concurrent.futures import ThreadPoolExecutor
import pandas as pd

def create_wandb_run_name(args, experiment_type="report"):
    """
    Generate a descriptive name for the wandb run based on input arguments.
    
    Args:
        args (argparse.Namespace): Parsed command-line arguments.
    
    Returns:
        str: A formatted name for the wandb run.
    """
    if experiment_type == "report":
        # Determine the report type
        if args.findings_only:
            report_type = "findings"
        elif args.impression_only:
            report_type = "impressions"
        elif args.combined:
            report_type = "combined"
        else:
            report_type = "full_report"
    
        # Base name components
        name_parts = [
            args.dataset, 
            report_type,                         # Report type: findings/impressions/combined/full_report
            "zeroshot",                          # Always include 'zeroshot' to identify the experiment type
            args.disease,                        # Disease being analyzed
            f"n{args.nr_reports_per_disease}",   # Number of reports per disease
        ]
        # Add dataset comparison tag if applicable
        if args.compare_to_mimic:
            name_parts.append("compare_to_mimic")
    elif experiment_type == "zeroshot":
        # Base name components
        name_parts = [
            args.dataset,                        # Dataset being analyzed
            "zeroshot",                          # Always include 'zeroshot' to identify the experiment type
            args.disease,   # Number of reports per disease
        ]

    elif experiment_type == "knn":
        # Base name components
        name_parts = [
            args.dataset,                        # Dataset being analyzed
            str(args.k_neighbors),                          # Always include 'zeroshot' to identify the experiment type
            "knn"   # Number of reports per disease
        ]
    elif experiment_type in ["mlp", "linear_probe", "lora"]:
        # Base name components
        name_parts = [
            args.dataset,                        # Dataset being analyzed
            experiment_type,   # Number of reports per disease
            str(args.train_data_percentage)]
        if args.train_vindr_percentage:
            name_parts.append("vindr_split")  

    elif experiment_type == "vte":
        # Base name components
        name_parts = [
            args.dataset,                        # Dataset being analyzed
            "vte"   # Number of reports per disease
        ]

    else:
        # Base name components
        name_parts = [
            args.dataset,                        # Dataset being analyzed
            "zeroshot",                          # Always include 'zeroshot' to identify the experiment type
            args.disease,   # Number of reports per disease
        ]
        
    if args.only_no_finding:
        name_parts.append("no_finding")
        
    # Add single disease filter flag
    if args.single_disease:
        name_parts.append("single_disease")
    
    
    # Join parts with underscores
    return "_".join(name_parts)



def read_image(image_path):
    with open(image_path, "rb") as f:
        return f.read()



def zero_shot_prediction(image_embedding, report_embeddings, report_labels, k=5):
    """
    Predict using a zero-shot approach by finding the k closest reports.
    
    Args:
    - image_embedding: Embedding of the input image (numpy array).
    - report_embeddings: Embeddings of the reference reports (numpy array).
    - report_labels: Labels of the reference reports (list or numpy array).
    - k: Number of closest reports to consider (int).
    
    Returns:
    - prediction: Predicted label (int, 0 for "No Finding", 1 for "Pneumonia").
    - closest_reports: List of indices of the k closest reports.
    """
    # Compute cosine similarity
    similarities = cosine_similarity(image_embedding.reshape(1, -1), report_embeddings).flatten()
    closest_indices = similarities.argsort()[-k:][::-1]  # Indices of top-k closest reports
    
    # Get labels of closest reports
    closest_labels = [report_labels[i] for i in closest_indices]
    
    # Predict based on majority label
    prediction = np.argmax(np.bincount(closest_labels))
    return prediction, closest_indices   
 


def extract_findings_and_impressions(reports, combine=False):
    """
    Extract findings and impressions from reports.

    Args:
        reports (list of str): List of report texts.
        combine (bool): If True, combines findings and impressions into one string.

    Returns:
        dict: A dictionary with keys 'findings', 'impressions', and optionally 'combined'.
    """
    findings_list = []
    impressions_list = []
    combined_list = [] if combine else None
    
    for report in reports:
        # Normalize spaces and line breaks
        report = " ".join(report.split())
        
        # Extract findings
        findings_match = re.search(r"FINDINGS:\s(.*?)(?=\sIMPRESSION:|\sCONCLUSION:|$)", report, re.IGNORECASE)
        findings = findings_match.group(1).strip() if findings_match else None
        findings = f"FINDINGS: {findings}" if findings else None
        findings_list.append(findings)
        
        # Extract impressions
        impression_match = re.search(r"IMPRESSION:\s(.*?)(?=\sFINDINGS:|$)", report, re.IGNORECASE)
        impression = impression_match.group(1).strip() if impression_match else None
        impression = f"IMPRESSION: {impression}" if impression else None
        impressions_list.append(impression)
        
        # Combine findings and impressions if requested
        if combine:
            if findings and impression:
                combined_list.append(f"{findings} {impression}")
            # elif findings:
            #     combined_list.append(findings)
            # elif impression:
            #     combined_list.append(impression)
            else:
                combined_list.append(None)
    
    result = {
        "findings": findings_list,
        "impressions": impressions_list
    }
    if combine:
        result["combined"] = combined_list
    
    return result

def calculate_subgroup_metrics(df_test, bias_variable, conditions, y_true, y_pred, y_prob, prompt_diseases):
    """
    Calculate metrics for each subgroup.
    """
    subgroup_metrics = {}
    for subgroup, condition in conditions.items():
        indices = df_test[condition(df_test)].index
        subgroup_y_true = [y_true[i] for i in indices if i < len(y_true)]
        subgroup_y_pred = [y_pred[i] for i in indices if i < len(y_pred)]
        subgroup_y_prob = [y_prob[i] for i in indices if i < len(y_prob)]

        if len(subgroup_y_true) == 0:
            continue

        accuracy = accuracy_score(subgroup_y_true, subgroup_y_pred)
        roc_auc = roc_auc_score(
            [[truth[disease] for disease in prompt_diseases] for truth in subgroup_y_true],
            [[prob[disease] for disease in prompt_diseases] for prob in subgroup_y_prob]
        ) if len(np.unique(subgroup_y_true)) > 1 else float('nan')

        subgroup_metrics[subgroup] = {
            "accuracy": accuracy,
            "roc_auc": roc_auc,
            "n_samples": len(subgroup_y_true),
        }

    return subgroup_metrics



def chest_xray_augmentations(image, num_views=3):
    """
    Generate a specified number of augmented views for a chest X-ray image in parallel.

    Args:
        image (PIL.Image): Input chest X-ray image (grayscale).
        num_views (int): Number of augmented views to generate.

    Returns:
        List[PIL.Image]: A list of augmented views.
    """
    # Define shape-preserving augmentations
    augmentation_transforms = [
        transforms.RandomRotation(degrees=10, fill=0),  # Small rotation
        transforms.RandomAffine(degrees=0, translate=(0.1, 0.1), fill=0),  # Translation
        transforms.ColorJitter(brightness=0.2, contrast=0.2),  # Brightness & contrast
        transforms.GaussianBlur(kernel_size=(5, 5)),  # Simulate blurring
        transforms.Lambda(lambda img: ImageOps.autocontrast(img)),  # Enhance contrast
    ]

    def apply_augmentation(_):
        # Randomly select and apply augmentations
        augment_pipeline = transforms.Compose(random.sample(augmentation_transforms, k=2))
        return augment_pipeline(image)

    # Use ThreadPoolExecutor for parallel augmentation
    with ThreadPoolExecutor() as executor:
        augmented_views = list(executor.map(apply_augmentation, range(num_views)))

    return augmented_views

def augment_image_to_base64(image, num_views=3):
    """
    Generate base64-encoded strings for augmented views of the input image in parallel.

    Args:
        image (PIL.Image): Input image.
        num_views (int): Number of augmented views to generate.

    Returns:
        List[str]: Base64-encoded strings of augmented images.
    """
    # Generate augmented views
    augmented_views = chest_xray_augmentations(image, num_views=num_views)
    all_views = [image] + augmented_views

    def encode_to_base64(augmented_image):
        # Save augmented image to an in-memory buffer as .jpg
        buffer = BytesIO()
        augmented_image.save(buffer, format="JPEG")
        buffer.seek(0)  # Move to the start of the buffer

        # Encode to base64
        base64_image = base64.encodebytes(buffer.read()).decode("utf-8")
        buffer.close()
        return base64_image

    # Use ThreadPoolExecutor for parallel base64 encoding
    with ThreadPoolExecutor() as executor:
        base64_encoded_images = list(executor.map(encode_to_base64, all_views))

    return base64_encoded_images

def select_confident_samples(logits, top):
    batch_entropy = -(logits.softmax(1) * logits.log_softmax(1)).sum(1)
    idx = torch.argsort(batch_entropy, descending=False)[:int(batch_entropy.size()[0] * top)]
    return idx

# Balance the dataset
def balance_dataset(df, disease = "Pneumonia", percentage = 1, vindr_samples = False):
    # get value count of df.disease and balance based on lowest value
    value_count = df[disease].value_counts()
    if not vindr_samples:
        minority_count = int(value_count.min()*percentage)
    else:
        vindr_split = {0.01: 6, 0.1: 74, 0.5:372, 0.8: 594, 1.0: 744}
        minority_count = int(vindr_split[percentage]/2)
        
    df_class_0 = df[df[disease] == 0].sample(minority_count, random_state=42)
    df_class_1 = df[df[disease] == 1].sample(minority_count, random_state=42)
    df_balanced = pd.concat([df_class_0, df_class_1])
    return df_balanced