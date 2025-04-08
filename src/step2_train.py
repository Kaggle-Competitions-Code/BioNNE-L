# -*- encoding: utf-8 -*-
'''
@create_time: 2025/03/28 10:37:16
@author: lichunyu
'''
import argparse
import os
from dataclasses import dataclass, field

import pandas as pd
import torch.nn as nn
from torch.nn import BCEWithLogitsLoss
from torch.utils.data import DataLoader, Dataset
from transformers import (
    AutoModel,
    BertForSequenceClassification,
    PreTrainedTokenizerBase,
)


class ELModel(nn.Module):

    def __init__(self, model_name_or_path: str, k: int, dropout: float = 0.1):
        super(ELModel, self).__init__()
        self.bert = AutoModel.from_pretrained(model_name_or_path)
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(self.bert.config.hidden_size, k)

    def forward(self, input_ids, attention_mask, token_type_ids, labels=None, type_labels=None):
        outputs = self.bert(input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)
        pooled_output = outputs[1]
        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)
        loss = None
        if labels is not None:
            loss_fct = BCEWithLogitsLoss()
            loss = loss_fct(logits, labels)
        if type_labels is not None:
            raise NotImplementedError()
        return {"logits": logits, "loss": loss}


class ELDataset(Dataset):

    def __init__(self):
        super().__init__()

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        return super().__getitem__(index)


@dataclass
class ELCollator:

    tokenizer: PreTrainedTokenizerBase = field(default=None, metadata={"help": "tokenizer of the LLM"})

    def __call__(self, *args, **kwds):
        pass


def main(args):
    train_data_path = args.train_data_path
    dev_data_path = args.dev_data_path

    train_data = pd.read_pickle(train_data_path)
    dev_data = pd.read_pickle(dev_data_path)
    ...


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_data_path",
                        type=str,
                        default="/media/f/lichunyu/BioNNE-L/data/eeyore/train_data.pkl",
                        help="Path to the training data")
    parser.add_argument("--dev_data_path",
                        type=str,
                        default="/media/f/lichunyu/BioNNE-L/data/eeyore/dev_data.pkl",
                        help="Path to the dev data")
    args = parser.parse_args()
    main(args)
    main(args)
