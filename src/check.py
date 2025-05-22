# -*- encoding: utf-8 -*-
'''
@create_time: 2025/04/18 09:21:29
@author: lichunyu
'''
import os

import pandas as pd
from datasets import load_dataset
from tqdm import tqdm


def split_sentence(text):
    sentences, sentence_ids = [], []
    chars = list(text)
    stack = ""
    for c_idx, c in enumerate(chars):
        if c == "." and c_idx + 1 < len(chars) and chars[c_idx + 1] == " " and stack != "":
            sentences.append(stack)
            sentence_ids.append((c_idx - len(stack), c_idx))
            stack = ""
        elif c != "." or (c == "." and c_idx + 1 < len(chars) and chars[c_idx + 1] != " "):
            if c == " " and stack == "":
                ...
            else:
                stack += c
    if stack != "":
        sentences.append(stack)
        sentence_ids.append((c_idx - len(stack), c_idx))
    return sentences, sentence_ids


def split_words(text, start_idx=0):
    words, word_ids = [], []
    chars = list(text)
    idx = 0
    stack = ""
    for c in chars:
        if c == " " and stack != "":
            words.append(stack)
            word_ids.append((start_idx + idx - len(stack), start_idx + idx))
            stack = ""
        elif c != " ":
            stack += c
        idx += 1
    if stack != "":
        words.append(stack)
        word_ids.append((start_idx + idx - len(stack), start_idx + idx))
    return words, word_ids


def get_span(start_idx, end_idx, num_prefix_words, num_suffix_words, doc_id, dataset_split, mention, base_doc_dir):
    start_idx = int(start_idx)
    end_idx = int(end_idx)
    doc_dir = os.path.join(base_doc_dir, doc_id[-2:], dataset_split)
    doc_path = os.path.join(doc_dir, doc_id + ".txt")
    with open(doc_path, "r") as f:
        text_with_spaces = f.read()
        text = text_with_spaces.replace("\n", " ")
        sentences, sentence_ids = split_sentence(text)
        for idx, (sentence_start_idx, sentence_end_idx) in enumerate(sentence_ids):
            if start_idx >= sentence_start_idx and end_idx <= sentence_end_idx:
                sentence_idx = idx
                break
        else:
            raise ValueError("Start and end indices are not in the same line")

        words, word_ids = split_words(sentences[sentence_idx], start_idx=sentence_start_idx)
        for idx, word in enumerate(words[:-1]):
            word_ids.append((word_ids[-1][0] + len(word) + 1, word_ids[-1][0] + len(word) + 1 + len(words[idx + 1])))

        for idx, (word_start_idx, word_end_idx) in enumerate(word_ids):
            if start_idx >= word_start_idx and start_idx <= word_end_idx:
                start = idx
            if end_idx <= word_end_idx and end_idx >= word_start_idx:
                end = idx

        start = max(0, start - num_prefix_words)
        end = min(end + num_suffix_words, len(words) - 1)
        span = " ".join(words[start:end + 1])

    print(f"Mention: {mention}")
    print(f"Span: {span}")
    return span


retrieval_batch_size = 600
df_en = load_dataset("andorei/BioNNE-L", "English", split="test").to_pandas()
df_ru = load_dataset("andorei/BioNNE-L", "Russian", split="test").to_pandas()
df = pd.concat([df_en, df_ru], axis=0)
# df = pd.read_pickle("/media/f/lichunyu/BioNNE-L/data/extra/pickle/ru/extra_ru_data.pkl")
num_error = 0
for chem_type in ("DISO", "CHEM", "ANATOMY"):
    sub_df = df[df["entity_type"] == chem_type]
    doc_ids = sub_df["document_id"].to_list()
    texts = sub_df["text"].to_list()
    spans = sub_df["spans"].to_list()
    CUIs = sub_df["UMLS_CUI"].to_list()
    for i in tqdm(range(0, len(doc_ids), retrieval_batch_size)):
        batch_doc_ids = doc_ids[i:i + retrieval_batch_size]
        batch_texts = texts[i:i + retrieval_batch_size]
        batch_spans = spans[i:i + retrieval_batch_size]
        batch_CUIs = CUIs[i:i + retrieval_batch_size]
        batch_span4train = []
        for j in range(len(batch_doc_ids)):
            start_idx, end_idx = batch_spans[j].split("-")[0], batch_spans[j].split("-")[-1]
            try:
                span_include_mention = get_span(
                    start_idx, end_idx, 50, 50, batch_doc_ids[j], "test", batch_texts[j],
                    "/media/f/lichunyu/BioNNE-L/data/NEREL-BIO/BioNNE-L_Shared_Task/data/texts")
            except Exception as e:
                num_error += 1
                span_include_mention = batch_texts[j]
            batch_span4train.append(span_include_mention)

...
