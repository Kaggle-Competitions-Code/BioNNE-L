# -*- encoding: utf-8 -*-
'''
@create_time: 2025/04/09 15:49:29
@author: lichunyu
'''
import argparse

import pandas as pd
import torch
import torch.nn.functional as F
from datasets import load_dataset
from tqdm import tqdm
from transformers import AutoModel, AutoTokenizer

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def get_umls():
    dataset = load_dataset("adlbh/umls-concepts", split="train")
    umls = dataset.to_pandas().to_dict("records")
    umls_map = {}
    for i in umls:
        umls_map[i["ENTITY"]] = i
    return umls_map


def main(args):
    model = AutoModel.from_pretrained(args.model_name_or_path).to(DEVICE)
    model.eval()
    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path)
    umls = get_umls()
    df = pd.read_pickle(args.data_path)
    num_acc = 0
    for idx, row in tqdm(df.iterrows(), total=len(df)):
        text = row["text"]
        mention = row["mention"]
        CUI = row["CUI"]
        candidates = row["candidates"]
        candidate_CUI = row["candidate_CUI"]
        labels = row["labels"]
        sentence = []
        for i in range(len(candidate_CUI)):
            # if candidate_CUI[i] in umls and umls[candidate_CUI[i]]["DEFINITION"] is not None:
            #     sentence.append(umls[candidate_CUI[i]]["DEFINITION"])
            # if candidate_CUI[i] in umls and umls[candidate_CUI[i]]["ALIASES"] is not None:
            #     sentence.append(umls[candidate_CUI[i]]["ALIASES"])
            # else:
            #     sentence.append(candidates[i])
            sentence.append(candidates[i])
        inputs = tokenizer(f"{mention}", padding="max_length", max_length=300, truncation=True, return_tensors="pt")
        inputs = {k: v.to(DEVICE) for k, v in inputs.items()}
        with torch.no_grad():
            outputs = model(**inputs)
            text_embedding = outputs.last_hidden_state[:, 0]
        inputs = tokenizer(sentence, padding="max_length", max_length=300, truncation=True, return_tensors="pt")
        inputs = {k: v.to(DEVICE) for k, v in inputs.items()}
        with torch.no_grad():
            outputs = model(**inputs)
            sentence_embeddings = outputs.last_hidden_state[:, 0]
        similarity = F.cosine_similarity(text_embedding, sentence_embeddings, dim=-1)
        score, ids = torch.topk(similarity, 1)
        if labels[ids[0]] == 1:
            num_acc += 1
        ...
    acc = num_acc / len(df)
    print(f"Acc: {acc:.4f}")
    ...


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name_or_path",
                        type=str,
                        default="/media/f/lichunyu/models/SapBERT-from-PubMedBERT-fulltext",
                        help="Path to the pretrained model")
    parser.add_argument("--data_path",
                        type=str,
                        default="/media/f/lichunyu/BioNNE-L/data/eeyore/train_data.pkl",
                        help="Path to the test data")
    args = parser.parse_args()
    main(args)
