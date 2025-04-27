# -*- encoding: utf-8 -*-
'''
@create_time: 2025/03/28 09:08:48
@author: lichunyu
'''
import argparse
import os
import random
from dataclasses import dataclass, field
from datetime import datetime

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from datasets import load_dataset
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, Dataset
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from transformers import (
    AutoModel,
    AutoTokenizer,
    BertForSequenceClassification,
    PreTrainedTokenizerBase,
    get_cosine_schedule_with_warmup,
)


class ELModel(nn.Module):

    def __init__(self, model_name_or_path: str, k: int, dropout: float = 0.2):
        super(ELModel, self).__init__()
        self.bert = AutoModel.from_pretrained(model_name_or_path)
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(self.bert.config.hidden_size, 2)

    def forward(self, input_ids, attention_mask, token_type_ids, labels=None, type_labels=None):
        outputs = self.bert(input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)
        pooled_output = outputs[1]
        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)
        loss = None
        if labels is not None:
            # loss_fct = BCEWithLogitsLoss()
            # loss = loss_fct(logits, labels)
            # loss_fct = MSELoss()
            # loss = loss_fct(logits.squeeze(), labels.squeeze())
            loss_fct = CrossEntropyLoss()
            loss = loss_fct(logits.view(-1, 2), labels.view(-1))
        if type_labels is not None:
            raise NotImplementedError()
        return {"logits": logits, "loss": loss}


class ELDataset(Dataset):

    def __init__(self, data, tokenizer, umls=None, max_length=512, k=10):
        self.data = data
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.k = k
        self.umls = umls if umls else {}

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        row = self.data.iloc[index]
        mention = row["mention"]
        text = row["text"]
        candidates = row["candidates"]
        candidate_CUI = row["candidate_CUI"]
        spans = row["spans"]
        doc_id = row["doc_id"]
        sentence0 = text.replace(mention, f"[Ms] {mention} [Me]")
        overall_inputs = {"input_ids": [], "attention_mask": [], "token_type_ids": []}
        for i in range(self.k):
            inputs = self.tokenizer(sentence0,
                                    candidates[i],
                                    max_length=self.max_length,
                                    truncation=True,
                                    padding=True)
            for k, v in inputs.items():
                overall_inputs[k].append(v)
        overall_inputs["candidate_CUI"] = candidate_CUI
        overall_inputs["spans"] = spans
        overall_inputs["doc_id"] = doc_id
        return overall_inputs


@dataclass
class ELCollator:

    tokenizer: PreTrainedTokenizerBase = field(default=None, metadata={"help": "tokenizer of the LLM"})

    def __call__(self, inputs):
        batch_size = len(inputs)
        input_ids = []
        attention_mask = []
        token_type_ids = []
        candidate_CUI = []
        spans = []
        doc_id = []
        for i in range(batch_size):
            input_ids.extend(inputs[i]["input_ids"])
            attention_mask.extend(inputs[i]["attention_mask"])
            token_type_ids.extend(inputs[i]["token_type_ids"])
            candidate_CUI.append(inputs[i]["candidate_CUI"])
            spans.append(inputs[i]["spans"])
            doc_id.append(inputs[i]["doc_id"])
        input_ids = pad_sequence([torch.tensor(i, dtype=torch.int) for i in input_ids], batch_first=True)
        attention_mask = pad_sequence([torch.tensor(i, dtype=torch.int) for i in attention_mask], batch_first=True)
        token_type_ids = pad_sequence([torch.tensor(i, dtype=torch.int) for i in token_type_ids], batch_first=True)
        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "token_type_ids": token_type_ids,
            "candidate_CUI": candidate_CUI,
            "spans": spans,
            "doc_id": doc_id,
        }


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def main(args):
    set_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = ELModel(args.model_name_or_path, args.retrieval_topk)
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_name_or_path)
    model.load_state_dict(torch.load(args.checkpoints_path))
    model.to(device)
    model.eval()
    data = pd.read_pickle(args.retrieval_data_path)
    dataset = ELDataset(data, tokenizer, k=args.retrieval_topk)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, collate_fn=ELCollator(tokenizer))
    data4df = {"document_id": [], "spans": [], "prediction": []}
    for batch in tqdm(dataloader):
        spans = batch["spans"]
        candidate_CUI = batch["candidate_CUI"]
        doc_id = batch["doc_id"]
        batch = {k: v.to(device) for k, v in batch.items() if isinstance(v, torch.Tensor)}
        with torch.no_grad():
            outputs = model(**batch)
            logits = outputs["logits"].detach().clone()
            pred = F.softmax(logits, dim=-1)[:, 1].view(-1, args.retrieval_topk).detach().cpu()
            pred = torch.argsort(pred, dim=-1, descending=True).numpy()
            for i in range(len(pred)):
                data4df["document_id"].append(doc_id[i])
                data4df["spans"].append(spans[i])
                data4df["prediction"].append([candidate_CUI[i][j]
                                              for j in pred[i]] + candidate_CUI[i][args.retrieval_topk:])
    df = pd.DataFrame(data4df)

    def prediction_func(prediction: list):
        result = []
        for pred in prediction:
            if pred not in result:
                result.append(pred)
        return result[:5]

    df["prediction"] = df["prediction"].apply(prediction_func)
    df = df.explode("prediction")
    df = df.reset_index(drop=False)
    df['rank'] = df.groupby('index').cumcount()
    df = df[["document_id", "spans", "rank", "prediction"]]
    df["rank"] = df["rank"] + 1
    df.to_csv(os.path.join(args.save_dir, args.save_file_name), index=False, sep="\t")
    print(f"Prediction file saved to {os.path.join(args.save_dir, args.save_file_name)}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--checkpoints_path",
        type=str,
        default=
        "/media/f/lichunyu/BioNNE-L/data/checkpoints/BiomedNLP-BiomedBERT-base-uncased-abstract-special_epoch_1_acc_0.6881.bin",
        help="")
    parser.add_argument("--model_name_or_path",
                        type=str,
                        default="/media/f/lichunyu/models/BiomedNLP-BiomedBERT-base-uncased-abstract-special",
                        help="")
    parser.add_argument("--tokenizer_name_or_path",
                        type=str,
                        default="/media/f/lichunyu/models/BiomedNLP-BiomedBERT-base-uncased-abstract-special",
                        help="")
    parser.add_argument("--retrieval_data_path",
                        type=str,
                        default="/media/f/lichunyu/BioNNE-L/data/eeyore/en/test_data_sap_50.pkl",
                        help="")
    parser.add_argument("--retrieval_topk", type=int, default=5, help="")
    parser.add_argument("--seed", type=int, default=42, help="")
    parser.add_argument("--batch_size", type=int, default=8, help="")
    parser.add_argument("--lang", type=str, default="en", help="")
    parser.add_argument("--save_file_name", type=str, default="test_en_predictions.tsv", help="")
    parser.add_argument("--save_dir", type=str, default="/media/f/lichunyu/BioNNE-L/data/eeyore/predictions", help="")
    args = parser.parse_args()
    main(args)
