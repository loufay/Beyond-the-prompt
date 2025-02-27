import os
import sys
import torch
import numpy as np
import pandas as pd
import wandb
from config import parse_arguments, get_dataset_config
from data_loader import load_data, prepare_samples
from metrics import evaluate_model, plot_confusion_matrix
current_dir = os.getcwd()
current_dir = current_dir + "/MedImageInsights"
sys.path.append(current_dir)
from MedImageInsight.medimageinsightmodel import MedImageInsight
from sklearn.metrics.pairwise import cosine_similarity
from utils import create_wandb_run_name, balance_dataset, evaluate_bias, zero_shot_prediction
from model.model import get_medimageinsight_classifier


def main():
    args = parse_arguments()
    dataset_config = get_dataset_config(args.dataset)
    bias_variables = dataset_config.get("bias_variables", None)

    # Initialize W&B
    run_name = create_wandb_run_name(args, "report")
    wandb.init(
        project=args.wandb_project,
        group=f"{args.dataset}-ZeroShot-Report",
        name=run_name,
    )

    output_dir = os.path.join(args.save_path, f"{args.dataset}-ZeroShot-Report", f"{wandb.run.group}_{wandb.run.name}")
    os.makedirs(output_dir, exist_ok=True)

    # Load and balance datasets
    data_path = os.path.join(os.getcwd(), "MedImageInsights/data", dataset_config["data_path"])
    _, df_val, df_test = load_data(data_path, args.disease)

    diseases_mimic = ['No Finding', 'Enlarged Cardiomediastinum', 'Cardiomegaly', 'Lung Opacity',
        'Lung Lesion', 'Edema', 'Consolidation', 'Pneumonia', 'Atelectasis',
        'Pneumothorax', 'Pleural Effusion', 'Pleural Other', 'Fracture',
        'Support Devices']
    
    read_path_mimic = os.path.join(os.getcwd(), "MedImageInsights/data/MIMIC-v1.0-512/")
    df_train = pd.read_csv(read_path_mimic + "train.csv")

    #d_test = df_test[(df_test["No Finding"] == 1) | (df_test[args.disease] == 1)]
    d_test = df_test[((df_test["No Finding"] == 1)  & (df_test[diseases_mimic[1:]]== 0).all(axis=1) )| (df_test[args.disease] == 1)]
    no_finding_samples_test = df_test[(df_test['No Finding'] == 1) & (df_test[diseases_mimic[1:]]== 0).all(axis=1)]
    finding_samples_test = df_test[(df_test[args.disease] == 1)]
    no_finding_samples_test = no_finding_samples_test.sample(len(finding_samples_test), random_state=42)
    
    df_test = pd.concat([finding_samples_test, no_finding_samples_test])

    df_test_balanced = balance_dataset(df_test, args.disease)

    classifier = get_medimageinsight_classifier()
    classifier.model.eval()

    # Compare to MIMIC reports
    nr_reports = 10

    # Sample Finding Reports
    finding_samples_train  = df_train[(df_train[args.disease] == 1)]
    no_finding_samples_train = df_train[(df_train['No Finding'] == 1) & (df_train[diseases_mimic[1:]]== 0).all(axis=1)]
    no_finding_samples_train = no_finding_samples_train.sample(len(finding_samples_train), random_state=42)

    if args.report_type == "Findings":
        no_finding_reports = no_finding_samples_train.section_findings.sample(nr_reports*2, random_state=42)
        finding_reports = finding_samples_train.section_findings.sample(nr_reports*2, random_state=42)
        # drop reports with no findings
        no_finding_reports = no_finding_reports.dropna()[0:nr_reports]
        finding_reports = finding_reports.dropna()[0:nr_reports]
    elif args.report_type == "Impression":
        no_finding_reports = no_finding_samples_train.section_impression.sample(nr_reports*2, random_state=42)
        finding_reports = finding_samples_train.section_impression.sample(nr_reports*2, random_state=42)
        # drop reports with no findings
        no_finding_reports = no_finding_reports.dropna()[0:nr_reports]
        finding_reports = finding_reports.dropna()[0:nr_reports]
    elif args.report_type == "Combined":
        no_finding_reports = no_finding_samples_train.section_findings.str.cat(no_finding_reports.section_impression, sep=" ", na_rep="").sample(nr_reports*2, random_state=42)
        finding_reports = finding_samples_train.section_findings.str.cat(finding_reports.section_impression, sep=" ", na_rep="").sample(nr_reports*2, random_state=42)
        # drop reports with no findings
        no_finding_reports = no_finding_reports.dropna()[0:nr_reports]
        finding_reports = finding_reports.dropna()[0:nr_reports]
    else:
        no_finding_reports = no_finding_samples_train.report.sample(nr_reports, random_state=42)
        finding_reports = finding_samples_train.report.sample(nr_reports, random_state=42)

      
    print(f"Number of No Finding Reports: {len(no_finding_reports)}")
    print(f"Number of {args.disease} Reports: {len(finding_reports)}")
    print(f"Number of Balanced Test data: {len(df_test_balanced)}")

    with torch.no_grad():
        report_texts = list(no_finding_reports) + list(finding_reports)
        report_embeddings = classifier.encode(texts=report_texts)["text_embeddings"]
    
    report_labels = [0]*len(no_finding_reports) + [1]*len(finding_reports)

    # Zero-Shot Prediction with k-NN Report Comparison
    all_predictions, all_labels, all_probabilities = [], [], []
    for i, row in df_test_balanced.iterrows():
        image_embedding = np.array(row[-1024:].values).reshape(1, 1024)

        # Zero-Shot Prediction using k-NN with Reports
        prediction, closest_indices, probabilities_based_on_labels = zero_shot_prediction(
            image_embedding=image_embedding, 
            report_embeddings=report_embeddings, 
            report_labels=report_labels, 
            k=5  # Number of closest reports to consider
        )

        # Use prediction and probabilities from k-NN
        true_label = 0 if row[args.disease] == 0 else 1
        all_probabilities.append(probabilities_based_on_labels)  
        all_predictions.append(prediction)
        all_labels.append(true_label)

    # Evaluate on Test Set using the provided evaluate_model function
    test_accuracy, test_auc, test_f1, test_cm = evaluate_model(
        probabilities=all_probabilities,
        predictions=all_predictions,
        labels=all_labels,
        dataset_name="Test",
        bias_variables=bias_variables,
        df=df_test_balanced
    )

    wandb.finish()
    

if __name__ == "__main__":
    main()




            


















