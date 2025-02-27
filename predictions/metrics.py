from sklearn.metrics import confusion_matrix, roc_auc_score, accuracy_score, f1_score, matthews_corrcoef
import matplotlib.pyplot as plt
import wandb
import os
import numpy as np
from sklearn.metrics import ConfusionMatrixDisplay


def evaluate_model(probabilities, predictions, labels, dataset_name, bias_variables=None, df=None):

    accuracy = accuracy_score(labels, predictions)
    auc = roc_auc_score(labels, probabilities)
    f1 = f1_score(labels, predictions, average="weighted")
    cm = confusion_matrix(labels, predictions)
    mcc = matthews_corrcoef(labels, predictions)

    no_findings_accuracy = cm[0, 0] / cm[0].sum() if cm[0].sum() > 0 else 0
    disease_accuracy = cm[1, 1] / cm[1].sum() if cm[1].sum() > 0 else 0

    print(f"{dataset_name} Accuracy: {accuracy:.4f}")
    print(f"{dataset_name} AUC: {auc:.4f}")
    print(f"{dataset_name} F1-Score: {f1:.4f}")
    print(f"{dataset_name} MCC: {mcc:.4f}")

    wandb.log({
        f"{dataset_name}_accuracy": accuracy,
        f"{dataset_name}_auc": auc,
        f"{dataset_name}_f1_score": f1,
        f"{dataset_name}_no_findings_accuracy": no_findings_accuracy,
        f"{dataset_name}_disease_accuracy": disease_accuracy,
        f"{dataset_name}_mcc": mcc,
    })


    if bias_variables and df is not None and dataset_name == "Test":
        evaluate_subgroup_metrics(labels, predictions, probabilities, bias_variables, df, dataset_name)

    return accuracy, auc, f1, cm

def plot_confusion_matrix(cm, dataset_name, labels):
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)
    disp.plot(cmap=plt.cm.Blues)
    plt.title(f"{dataset_name} Confusion Matrix")
    plt.savefig(os.path.join(output_dir, f"{dataset_name}_confusion_matrix.png"))
    wandb.log({f"{dataset_name}_confusion_matrix": wandb.Image(plt)})
    plt.close()

def evaluate_subgroup_metrics(y_true, y_pred, y_prob, bias_variables, df, dataset_name):
    print(f"\nEvaluating Subgroup Metrics for {dataset_name}...")
    for variable, conditions in bias_variables.items():
        print(f"\nAnalyzing bias for {variable}...")
        subgroup_metrics = {}

        for subgroup, condition in conditions.items():
            indices = df[condition(df)].index
            subgroup_y_true = [y_true[i] for i in indices if i < len(y_true)]
            subgroup_y_pred = [y_pred[i] for i in indices if i < len(y_pred)]
            subgroup_y_prob = [y_prob[i] for i in indices if i < len(y_prob)]

            accuracy = accuracy_score(subgroup_y_true, subgroup_y_pred)
            roc_auc = roc_auc_score(subgroup_y_true, subgroup_y_prob) if len(np.unique(subgroup_y_true)) > 1 else float('nan')
            mcc = matthews_corrcoef(subgroup_y_true, subgroup_y_pred)
            cm_subgroup = confusion_matrix(subgroup_y_true, subgroup_y_pred)
            no_findings_accuracy = cm_subgroup[0, 0] / cm_subgroup[0].sum() if cm_subgroup[0].sum() > 0 else 0
            findings_accuracy = cm_subgroup[1, 1] / cm_subgroup[1].sum() if cm_subgroup[1].sum() > 0 else 0

            subgroup_metrics[subgroup] = {
                "accuracy": accuracy,
                "roc_auc": roc_auc,
                "n_samples": len(subgroup_y_true),
                "no_findings_accuracy": no_findings_accuracy,
                "findings_accuracy": findings_accuracy,
                "mcc": mcc,
            }

        for subgroup, metrics in subgroup_metrics.items():
            print(f"{variable} - {subgroup}: Accuracy = {metrics['accuracy']:.4f}, AUC = {metrics['roc_auc']:.4f}")
            wandb.log({
                f"{dataset_name}_{variable}_{subgroup}_accuracy": metrics["accuracy"],
                f"{dataset_name}_{variable}_{subgroup}_roc_auc": metrics["roc_auc"],
                f"{dataset_name}_{variable}_{subgroup}_n_samples": metrics["n_samples"],
                f"{dataset_name}_{variable}_{subgroup}_no_findings_accuracy": metrics["no_findings_accuracy"],
                f"{dataset_name}_{variable}_{subgroup}_findings_accuracy": metrics["findings_accuracy"],
                f"{dataset_name}_{variable}_{subgroup}_mcc": metrics["mcc"],
            })