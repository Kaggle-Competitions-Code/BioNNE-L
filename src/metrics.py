# -*- encoding: utf-8 -*-
'''
@create_time: 2025/04/09 11:34:35
@author: lichunyu
'''
import argparse

import pandas as pd


def compute_metrics_step1(df):
    num_data = len(df)
    acc, acc_2, acc_5, acc_10 = 0, 0, 0, 0
    for idx, row in df.iterrows():
        labels = row["labels"]
        if labels[0] == 1:
            acc += 1
        if 1 in labels[:2]:
            acc_2 += 1
        if 1 in labels[:5]:
            acc_5 += 1
        if 1 in labels[:10]:
            acc_10 += 1
    acc = acc / num_data
    acc_2 = acc_2 / num_data
    acc_5 = acc_5 / num_data
    acc_10 = acc_10 / num_data
    print(f"Acc: {acc:.4f}, Acc@2: {acc_2:.4f}, Acc@5: {acc_5:.4f}, Acc@10: {acc_10:.4f}")
    return acc, acc_5, acc_10


def main(args):
    df = pd.read_pickle(args.data_path)
    if args.step == 1:
        metrics = compute_metrics_step1(df)
    elif args.step == 2:
        ...


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--step", type=int, default=1, help="Step number")
    parser.add_argument("--data_path",
                        type=str,
                        default="/media/f/lichunyu/BioNNE-L/data/eeyore/dev_data_sap_50.pkl",
                        help="Path to the test data")
    args = parser.parse_args()
    main(args)
