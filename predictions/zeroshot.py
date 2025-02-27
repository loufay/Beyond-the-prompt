import os
import torch
import numpy as np
import pandas as pd
import wandb
from config import parse_arguments, get_dataset_config
from data_loader import load_data, prepare_samples
from metrics import evaluate_model, plot_confusion_matrix
from sklearn.metrics.pairwise import cosine_similarity
import sys
current_dir = os.getcwd()
current_dir = current_dir + "/MedImageInsights"
sys.path.append(current_dir)
from MedImageInsight.medimageinsightmodel import MedImageInsight
from utils import create_wandb_run_name, balance_dataset, evaluate_bias
from model.model import get_medimageinsight_classifier
from tqdm import tqdm



def main():
    args = parse_arguments()
    dataset_config = get_dataset_config(args.dataset)
    bias_variables = dataset_config.get("bias_variables", None)

    # Initialize W&B
    base_run_name = create_wandb_run_name(args, "zeroshot")

    # Load and balance datasets
    data_path = os.path.join(os.getcwd(), "MedImageInsights/data", dataset_config["data_path"])
    df_train, df_val, df_test = load_data(data_path, args.disease)
    df_test_balanced = balance_dataset(df_test, args.disease)

    # print number of samples
    print(f"Test samples: {len(df_test_balanced)}")

    classifier = get_medimageinsight_classifier()
    classifier.model.eval()

    add_ons = ["x-ray chest anteroposterior ", "x-ray chest ", "x-ray ", "chest ", ""]

    for add_on in add_ons:
        # Start a new W&B run for each add_on
        run_name = f"{base_run_name}_{add_on.strip().replace(' ', '_')}"
        wandb.init(
            project="MedImageInsights_7",
            group=f"{args.dataset}-ZeroShot",
            name=run_name,
            reinit=True
        )

        save_path = os.path.join(args.save_path, args.dataset, run_name)
        os.makedirs(save_path, exist_ok=True)

        print(f"Processing >{add_on}< dataset")
        prompt_diseases = [f"{add_on}{class_name}" for class_name in ["No Finding", args.disease]]

        with torch.no_grad():
            prompt_embeddings = classifier.encode(texts=prompt_diseases)["text_embeddings"]

        ground_truth, all_predictions, all_probabilities = [], [], []


        for i, row in tqdm(df_test_balanced.iterrows(), total=len(df_test_balanced), desc="Zeroshot performance"):
            image_embedding = np.array(row[-1024:].values).reshape(1, 1024)
            similarity = cosine_similarity(image_embedding.reshape(1, -1), prompt_embeddings).flatten()
            probabilities = np.exp(similarity) / np.exp(similarity).sum()
            all_probabilities.append(probabilities[1])
            y_pred = 1 if probabilities[1] > 0.5 else 0
            y_true = row[args.disease]
            all_predictions.append(y_pred)
            ground_truth.append(y_true)

        # Evaluate on Test Set using the provided evaluate_model function
        test_accuracy, test_auc, test_f1, test_cm = evaluate_model(
            probabilities=all_probabilities,
            predictions=all_predictions,
            labels=ground_truth,
            dataset_name="Test",
            bias_variables=bias_variables,
            df=df_test_balanced
        )

        wandb.finish()


if __name__ == "__main__":
    main()