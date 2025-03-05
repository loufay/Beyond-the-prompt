
import argparse

def parse_arguments():
    parser = argparse.ArgumentParser(description="Model training and evaluation")
    parser.add_argument("--dataset", type=str, default="CheXpert", help="Dataset to use (MIMIC, CheXpert, VinDR)")
    parser.add_argument("--save_path", type=str, default="./Results/", help="Path to save the results")
    parser.add_argument("--only_no_finding", action="store_false", help="Filter reports for 'No Finding' samples")
    parser.add_argument("--single_disease", action="store_true", help="Filter reports for single disease occurrence")
    parser.add_argument("--train_data_percentage", type=float, default=1.0, help="Percentage of training data to use")
    parser.add_argument("--train_vindr_percentage", action="store_false", help="Percentage of training data to use")
    parser.add_argument("--disease", type=str, default="Pneumonia", help="Disease to analyze")
    parser.add_argument("--weights", type=str, default="None", help="Weight function used in prediction")

    # Path to data
    parser.add_argument("--data_path", type=str, default="./data", help="Path to the data")
    # kNN
    parser.add_argument("--k_neighbors", type=int, default=5, help="Number of neighbors for k-NN")

    # Training
    parser.add_argument("--max_epochs", type=int, default=100, help="Maximum number of epochs for training")

    # LoRA
    parser.add_argument("--rank", type=int, default=8, help="Rank of the tensor decomposition")

    # Zeroshot Reports
    parser.add_argument("--report_type", type=str, default="Findings", help="Findings, Impression, Combined, or Full Report")

    # W&B configuration
    parser.add_argument("--wandb_project", type=str, default="MedImageInsights_5", help="Weights & Biases project name")

    # VTE
    parser.add_argument("--image_processing", type=str, default="avg_all", help="Image processing method [original, avg_all, avg_confidence]")
    parser.add_argument("--text_processing", type=str, default="all", help="Text processing method [[all, prompts_only, reports_only, weighted_all, weighted_prompts_only, weighted_reports_only]")
    
    # BioMedClip
    parser.add_argument("--model_type", type=str, default="BioMedCLIP", help="Model type [MedImageInsight, BioMedCLIP]")
    return parser.parse_args()



def get_dataset_config(dataset):
    configs = {
        "MIMIC": {
            "data_path": "MIMIC-v1.0-512",
            "diseases": ['No Finding', 'Enlarged Cardiomediastinum', 'Cardiomegaly', 'Lung Opacity', 'Lung Lesion', 'Edema', 'Consolidation', 'Pneumonia', 'Atelectasis', 'Pneumothorax', 'Pleural Effusion', 'Pleural Other', 'Fracture', 'Support Devices'],
            "bias_variables": None
        },
        "CheXpert": {
            "data_path": "CheXpert-v1.0-512",
            "diseases": ['No Finding', 'Enlarged Cardiomediastinum', 'Cardiomegaly', 'Lung Opacity', 'Lung Lesion', 'Edema', 'Consolidation', 'Pneumonia', 'Atelectasis', 'Pneumothorax', 'Pleural Effusion', 'Pleural Other', 'Fracture', 'Support Devices'],
        "bias_variables":{
            "sex": {"Female": lambda df: df["sex"] == "Female", "Male": lambda df: df["sex"] == "Male"},
            "age": {"Young": lambda df: df["age"] <= 62, "Old": lambda df: df["age"] > 62},
            "race": {
                "White": lambda df: df["race"] == "White",
                "Asian": lambda df: df["race"] == "Asian",
                "Black": lambda df: df["race"] == "Black",
                }
            },
        },
        "VinDR": {
            "data_path": "vindr-pcxr",
            "diseases": ['No Finding', 'Bronchitis', 'Brocho-pneumonia', 'Other disease', 'Bronchiolitis', 'Situs inversus', 'Pneumonia', 'Pleuro-pneumonia', 'Diagphramatic hernia', 'Tuberculosis', 'Congenital emphysema', 'CPAM', 'Hyaline membrane disease', 'Mediastinal tumor', 'Lung tumor'],
            "bias_variables": None

        }

    }
    return configs.get(dataset, {})
