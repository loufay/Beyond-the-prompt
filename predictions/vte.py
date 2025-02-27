import os
import sys
import torch
import numpy as np
import pandas as pd
import wandb
import matplotlib.pyplot as plt
import seaborn as sns
from config import parse_arguments, get_dataset_config
from data_loader import load_data, prepare_samples
from metrics import evaluate_model
from sklearn.metrics import confusion_matrix, roc_auc_score, accuracy_score, matthews_corrcoef
from sklearn.metrics.pairwise import cosine_similarity
from tqdm import tqdm
from PIL import Image

current_dir = os.getcwd()
current_dir = current_dir + "/MedImageInsights"
sys.path.append(current_dir)

from MedImageInsight.medimageinsightmodel import MedImageInsight
from utils import (
    read_image, 
    zero_shot_prediction, 
    create_wandb_run_name, 
    augment_image_to_base64, 
    select_confident_samples
)
from model.model import get_medimageinsight_classifier
from model.weighted_prompts import extract_weighted_text_embeddings

def main():
    
    args = parse_arguments()
    dataset_config = get_dataset_config(args.dataset)
    bias_variables = dataset_config.get("bias_variables", None)
    PATH_TO_DATA = os.path.join(os.getcwd(), "MedImageInsights/data")


    run_name = create_wandb_run_name(args, experiment_type="vte")
    wandb.init(
        project=args.wandb_project,
        group=f"{args.dataset}-vte",
        name=run_name,
    )

    data_path = os.path.join(PATH_TO_DATA, dataset_config["data_path"])

    df_test = pd.read_csv(data_path + "/test.csv")
    df_test = df_test.sample(frac=1, random_state=42).reset_index(drop=True) # Shuffle the dataset

    sample_size = min(df_test[args.disease].value_counts().values)
    df_test_no_finding = df_test[df_test["No Finding"] == 1].sample(n=sample_size, random_state=42)
    df_test_disease = df_test[df_test[args.disease] == 1].sample(n=sample_size, random_state=42)
    df_test = pd.concat([df_test_no_finding, df_test_disease]).reset_index(drop=True)
    print(f"Test dataset size: {len(df_test)}")
    print(f"No Finding: {len(df_test_no_finding)}")
    print(f"{args.disease}: {len(df_test_disease)}")


    if args.text_processing == "all":
        averaged_text_embeddings = np.load(f"{PATH_TO_DATA}/text_embeddings/average_embeddings_2_no_finding.npy")
    elif args.text_processing == "max_all":
        averaged_text_embeddings = np.load(f"{PATH_TO_DATA}/text_embeddings/max_embeddings_2_no_finding.npy")
    elif args.text_processing == "prompts_only":
        averaged_text_embeddings = np.load(f"{PATH_TO_DATA}/text_embeddings/average_embeddings_2_no_finding_template.npy")
    elif args.text_processing == "reports_only":
        averaged_text_embeddings = np.load(f"{PATH_TO_DATA}/text_embeddings/average_embeddings_2_no_finding_report.npy")
    elif "weighted" in args.text_processing:
        averaged_text_embeddings = extract_weighted_text_embeddings(args)
    else:
        raise ValueError("Invalid text processing method")

    classifier = get_medimageinsight_classifier()
    classifier.model.eval()

    all_predictions, ground_truth, all_logits = [], [], []

    
    for idx, row in tqdm(df_test.iterrows(),total=len(df_test), desc="Zeroshot performance"):

        # Create augmented image views and encode to base64
        if args.image_processing == "original":
            augmented_image_embeddings =  np.array(row[-1024:].values).reshape(1, 1024)
        else:
            path_to_img =PATH_TO_DATA+row["Path"]
            img = Image.open(path_to_img).convert("L")
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
        
        
        final_scores = cosine_similarity([average_filtered_embedding], averaged_text_embeddings)  # Shape: (1, K)
        # Apply softmax for probabilities as torch tensor
        final_probabilities = torch.nn.functional.softmax(torch.tensor(final_scores), dim=1)

        # Get the predicted class
        y_true = int(row[args.disease])
        y_pred = final_probabilities.argmax(dim=1).item()

        all_logits.append(final_probabilities[:, 1].item()) 
        all_predictions.append(y_pred)
        ground_truth.append(y_true)


    test_accuracy, test_auc, test_f1, test_cm = evaluate_model(
        probabilities=all_logits,
        predictions=all_predictions,
        labels=ground_truth,
        dataset_name="Test",
        bias_variables=None,
        df=df_test
    )

    wandb.finish()

if __name__ == "__main__":
    main()



    
