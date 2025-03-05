import io
import os
import requests
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers import AutoModelForCausalLM, AutoProcessor, GenerationConfig
from model import CheXagent
import sys
current_dir = os.getcwd()
current_dir = current_dir + "/MedImageInsights/predictions"
sys.path.append(current_dir)
from metrics import evaluate_model
from utils import create_wandb_run_name, balance_dataset, evaluate_bias
import wandb
from tqdm import tqdm
from data_loader import load_data, prepare_samples


# huggingface-cli login
# if other gpu set torch.float16 back to torch.bfloat16

from config import parse_arguments, get_dataset_config
 

def main():
    args = parse_arguments()
    dataset_config = get_dataset_config(args.dataset)
    bias_variables = dataset_config.get("bias_variables", None)

    PATH_TO_DATA = os.path.join(os.getcwd(), "MedImageInsights/data")
    
    # Initialize W&B
    run_name = create_wandb_run_name(args, "cheXagent")

    wandb.init(
        project=args.wandb_project,
        group=f"{args.dataset}-CheXagent",
        name=run_name,
    )  

    # Load and balance datasets
    data_path = os.path.join(os.getcwd(), "MedImageInsights/data", dataset_config["data_path"])
    df_train, df_val, df_test = load_data(data_path, args.disease)
    df_test_balanced = balance_dataset(df_test, args.disease) 
    
    
    # print number of samples
    print(f"Test samples: {len(df_test_balanced)}")

    chexagent = CheXagent()

    diseases = [args.disease]

    all_predictions, all_labels = [], []
    for i, row in tqdm(df_test_balanced.iterrows(), total=len(df_test_balanced), desc="CheXagent performance"):
        path_to_img =PATH_TO_DATA+row["Path"]

        response = chexagent.binary_disease_classification([path_to_img], args.disease)

        pred = 1 if args.disease.lower() in response.lower() else 0
        all_predictions.append(pred)
        all_labels.append(row[args.disease])
    
    # Evaluate the model
    test_accuracy, test_auc, test_f1, test_cm = evaluate_model(
            probabilities=None,
            predictions=all_predictions,
            labels=all_labels,
            dataset_name="Test",
            bias_variables=bias_variables,
            df=df_test_balanced
        )
    
    wandb.finish()


if __name__ == "__main__":
    main()


