# -*- encoding: utf-8 -*-
'''
@create_time: 2025/04/16 09:13:41
@author: lichunyu
'''
import argparse
import os

import pandas as pd
from datasets import load_dataset

DATA_DIR = "/media/f/lichunyu/BioNNE-L/data/eeyore"


def compute_metirc(df, dataset):
    dataset = dataset.to_pandas()
    num_data = len(dataset)
    num_correct = 0
    for i in range(len(dataset)):
        if df[(df["document_id"] == dataset.loc[i]["document_id"])
              & (df["spans"] == dataset.loc[i]["spans"])].iloc[0]["prediction"] == dataset.loc[i]["UMLS_CUI"]:
            num_correct += 1
    print(f"Acc@1: {num_correct / num_data:.4f}")


def main(args):
    df = pd.read_pickle(os.path.join(DATA_DIR, args.lang, args.data_path))
    # dataset = load_dataset("andorei/BioNNE-L", "Russian", split="dev")
    df = df[["doc_id", "spans", "candidate_CUI"]].rename(columns={
        "doc_id": "document_id",
        "candidate_CUI": "prediction"
    })

    def prediction_func(prediction: list):
        result = []
        for pred in prediction:
            if pred not in result:
                result.append(pred)
        return result[:5]

    df["prediction"] = df["prediction"].apply(prediction_func)
    df = df.explode("prediction")
    # compute_metirc(df, dataset)
    df = df.reset_index(drop=False)
    df['rank'] = df.groupby('index').cumcount()
    df = df[["document_id", "spans", "rank", "prediction"]]
    df["rank"] = df["rank"] + 1
    df.to_csv(os.path.join(DATA_DIR, "predictions", args.save_file_name), index=False, sep="\t")
    print(f"Prediction file saved to {args.save_file_name}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str, default="test_data_sap_50.pkl", help="Path to the test data")
    parser.add_argument("--lang", type=str, default="bilingual", help="")
    parser.add_argument("--save_file_name", type=str, default="test_bilingual_predictions.tsv", help="")
    args = parser.parse_args()
    main(args)
