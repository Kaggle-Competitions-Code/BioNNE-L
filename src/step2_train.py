# -*- encoding: utf-8 -*-
'''
@create_time: 2025/03/28 10:37:16
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
        CUI = row["CUI"]
        candidates = row["candidates"]
        candidate_CUI = row["candidate_CUI"]
        labels = row["labels"]
        sentence0 = text.replace(mention, f"[Ms] {mention} [Me]")
        # sentence1 = ""
        # sentence0 = text
        overall_inputs = {"input_ids": [], "attention_mask": [], "token_type_ids": []}
        for i in range(self.k):
            inputs = self.tokenizer(
                sentence0,
                # self.umls[candidate_CUI[i]] if candidate_CUI[i] in self.umls
                # and self.umls[candidate_CUI[i]] else candidates[i],
                candidates[i],
                max_length=self.max_length,
                truncation=True,
                padding=True)
            for k, v in inputs.items():
                overall_inputs[k].append(v)
        overall_inputs["label"] = labels[:self.k]
        return overall_inputs


@dataclass
class ELCollator:

    tokenizer: PreTrainedTokenizerBase = field(default=None, metadata={"help": "tokenizer of the LLM"})

    def __call__(self, inputs):
        batch_size = len(inputs)
        input_ids = []
        attention_mask = []
        token_type_ids = []
        labels = []
        for i in range(batch_size):
            input_ids.extend(inputs[i]["input_ids"])
            attention_mask.extend(inputs[i]["attention_mask"])
            token_type_ids.extend(inputs[i]["token_type_ids"])
            labels.extend(inputs[i]["label"])
        input_ids = pad_sequence([torch.tensor(i, dtype=torch.int) for i in input_ids], batch_first=True)
        attention_mask = pad_sequence([torch.tensor(i, dtype=torch.int) for i in attention_mask], batch_first=True)
        token_type_ids = pad_sequence([torch.tensor(i, dtype=torch.int) for i in token_type_ids], batch_first=True)
        labels = torch.tensor(labels)
        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "token_type_ids": token_type_ids,
            "labels": labels
        }


def compute_metrics(labels, logits):
    ...


def get_umls():
    dataset = load_dataset("adlbh/umls-concepts", split="train")
    umls = dataset.to_pandas().to_dict("records")
    umls_map = {}
    for i in umls:
        umls_map[i["ENTITY"]] = i["ALIASES"]
    return umls_map


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

    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_name_or_path)

    train_data = pd.read_pickle(args.train_data_path)
    dev_data = pd.read_pickle(args.dev_data_path)

    # umls = get_umls()

    train_dataset = ELDataset(train_data, tokenizer, max_length=args.max_length, k=args.retrieval_topk)
    dev_dataset = ELDataset(dev_data, tokenizer, max_length=args.max_length, k=args.retrieval_topk)
    train_dataloader = DataLoader(train_dataset,
                                  batch_size=args.train_batch_size,
                                  collate_fn=ELCollator(tokenizer),
                                  shuffle=True)
    dev_dataloader = DataLoader(dev_dataset, batch_size=args.eval_batch_size, collate_fn=ELCollator(tokenizer))

    model = ELModel(args.model_name_or_path, k=args.retrieval_topk)
    model.to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate)
    # scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=2)
    scheduler = get_cosine_schedule_with_warmup(optimizer=optimizer,
                                                num_warmup_steps=20,
                                                num_training_steps=args.num_train_epochs * len(train_dataloader))

    writer = SummaryWriter(log_dir=os.path.join(args.log_dir, datetime.now().strftime("%Y-%m-%d_%H-%M-%S")))

    for epoch in tqdm(range(args.num_train_epochs)):
        model.train()
        for step, batch in tqdm(enumerate(train_dataloader), total=len(train_dataloader)):
            batch = {k: v.to(device) for k, v in batch.items()}
            outputs = model(**batch)
            loss = outputs["loss"]
            loss.backward()
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()
            writer.add_scalar("train/loss", loss.item(), epoch * len(train_dataloader) + step)

        # Evaluation
        model.eval()
        num_all, num_correct = 0, 0
        for step, batch in tqdm(enumerate(dev_dataloader), total=len(dev_dataloader)):
            batch = {k: v.to(device) for k, v in batch.items()}
            with torch.no_grad():
                outputs = model(**batch)
                loss = outputs["loss"]
                writer.add_scalar("dev/loss", loss.item(), epoch * len(dev_dataloader) + step)
                logits = outputs["logits"].detach().clone()
                pred = F.softmax(logits)[:, 1].view(-1, args.retrieval_topk).detach().cpu().numpy()
                # logits = outputs["logits"].view(-1, args.retrieval_topk).detach().cpu().numpy()
                pred = np.argmax(pred, axis=-1)
                labels = batch["labels"].view(-1, args.retrieval_topk).detach().cpu().numpy()
                for idx, i in enumerate(pred):
                    if labels[idx][i] == 1:
                        num_correct += 1
                    num_all += 1
        acc = num_correct / num_all
        writer.add_scalar("dev/acc", acc, epoch)
        torch.save(
            model.state_dict(),
            f"{os.path.join(args.save_dir, os.path.basename(args.model_name_or_path))}_epoch_{epoch}_acc_{acc:.4f}.bin"
        )


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_data_path",
                        type=str,
                        default="/media/f/lichunyu/BioNNE-L/data/eeyore/en/train_data_sap_50.pkl",
                        help="Path to the training data")
    parser.add_argument("--dev_data_path",
                        type=str,
                        default="/media/f/lichunyu/BioNNE-L/data/eeyore/en/dev_data_sap_50.pkl",
                        help="Path to the dev data")
    parser.add_argument("--model_name_or_path",
                        type=str,
                        default="/media/f/lichunyu/models/BiomedNLP-BiomedBERT-base-uncased-abstract-fulltext-special",
                        help="Path to the pretrained model")
    parser.add_argument("--tokenizer_name_or_path",
                        type=str,
                        default="/media/f/lichunyu/models/BiomedNLP-BiomedBERT-base-uncased-abstract-fulltext-special",
                        help="Path to the pretrained tokenizer")
    parser.add_argument("--max_length", type=int, default=500, help="Maximum length of the input sequences")
    parser.add_argument("--train_batch_size", type=int, default=16, help="Batch size for training")
    parser.add_argument("--eval_batch_size", type=int, default=16, help="Batch size for evaluation")
    parser.add_argument("--retrieval_topk", type=int, default=5, help="Top k retrieval results")
    parser.add_argument("--learning_rate", type=float, default=7e-6, help="Learning rate for the optimizer")
    parser.add_argument("--num_train_epochs", type=int, default=10, help="Number of training epochs")
    parser.add_argument("--log_dir",
                        type=str,
                        default="/media/f/lichunyu/BioNNE-L/data/runs",
                        help="Directory to save logs and checkpoints")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for initialization")
    parser.add_argument("--save_dir",
                        type=str,
                        default="/media/f/lichunyu/BioNNE-L/data/checkpoints",
                        help="Directory to save checkpoints")
    args = parser.parse_args()
    main(args)
