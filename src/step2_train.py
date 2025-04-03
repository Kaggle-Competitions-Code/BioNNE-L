# -*- encoding: utf-8 -*-
'''
@create_time: 2025/03/28 10:37:16
@author: lichunyu
'''
import torch.nn as nn
from transformers import AutoModel, BertForSequenceClassification


class ELModel(nn.Module):

    def __init__(self, model_name_or_path: str, k: int, dropout: float = 0.1):
        super(ELModel, self).__init__()
        self.bert = AutoModel.from_pretrained(model_name_or_path)
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(self.bert.config.hidden_size, k)

    def forward(self,
                input_ids,
                attention_mask,
                token_type_ids,
                labels=None,
                type_labels=None):
        outputs = self.bert(input_ids,
                            attention_mask=attention_mask,
                            token_type_ids=token_type_ids)
        pooled_output = outputs[1]
        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)
        loss = None
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
        if type_labels is not None:
            raise NotImplementedError()
        return logits


def main():
    ...


if __name__ == '__main__':
    ...
