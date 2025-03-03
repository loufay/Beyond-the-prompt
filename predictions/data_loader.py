import pandas as pd
from utils import balance_dataset
import numpy as np

def load_data(data_path, disease):
    df_train = pd.read_csv(f"{data_path}/train.csv")
    df_val = pd.read_csv(f"{data_path}/val.csv")
    df_test = pd.read_csv(f"{data_path}/test.csv")

    df_train = df_train[(df_train["No Finding"] == 1) | (df_train[disease] == 1)]
    df_val = df_val[(df_val["No Finding"] == 1) | (df_val[disease] == 1)]
    df_test = df_test[(df_test["No Finding"] == 1) | (df_test[disease] == 1)]

    return df_train, df_val, df_test

# Prepare features and labels
def prepare_data(df, dataset_name="Train", disease="Pneumonia"):
    features = np.vstack(df[df.columns[-1024:]].values)
    labels = df[disease].values
    # print number of samples
    print(f"{dataset_name} samples: {len(labels)}")
    return features, labels

# Prepare samples
def prepare_samples(df, feature_columns, disease):
    return {
        "img_name": df["Path"].tolist(),
        "labels": df[disease].tolist(),
        "features": [np.array(row) for row in df[feature_columns].values]
    }