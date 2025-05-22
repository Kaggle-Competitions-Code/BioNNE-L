# -*- encoding: utf-8 -*-
'''
@create_time: 2025/04/09 09:57:48
@author: lichunyu
'''
import argparse
import os

from transformers import AutoModel, AutoTokenizer


def add_special_token(args):
    path = args.model_name_or_path
    special_tokens = ["[Ms]", "[Me]"]

    tokenizer = AutoTokenizer.from_pretrained(path)
    model = AutoModel.from_pretrained(path)
    tokenizer.add_tokens(special_tokens)
    model.resize_token_embeddings(len(tokenizer))
    tokenizer.save_pretrained(f"{path}-special")
    model.save_pretrained(f"{path}-special")
    print("add special token done")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name_or_path",
                        type=str,
                        default="/media/f/lichunyu/models/SapBERT-UMLS-2020AB-all-lang-from-XLMR-large",
                        help="Path to the pretrained model")
    args = parser.parse_args()
    add_special_token(args)

...
