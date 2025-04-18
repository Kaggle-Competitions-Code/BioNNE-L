# -*- encoding: utf-8 -*-
'''
@create_time: 2025/03/31 17:04:34
@author: lichunyu
'''
import argparse
import os
import warnings
from collections import defaultdict

import pandas as pd
import torch
import torch.nn.functional as F
from datasets import load_dataset
from tqdm import tqdm
from transformers import AutoModel, AutoTokenizer

warnings.filterwarnings("ignore")

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

DOC_DIR = "/media/f/lichunyu/BioNNE-L/data/NEREL-BIO/BioNNE-L_Shared_Task/data/texts"

DATA_DIR = "/media/f/lichunyu/BioNNE-L/data/eeyore"


def filter_vocab(vocab, lang):
    if lang == "ru":
        ru_cuis_set = set(vocab[vocab["lang"] == "RUS"]["CUI"].unique())
        vocab = vocab[vocab["CUI"].isin(ru_cuis_set)][["CUI", "semantic_type", "concept_name"]]
        print(f"Created Russian vocab: {vocab.shape}")
    elif lang == "en":
        vocab = vocab[vocab["lang"] == "ENG"][["CUI", "semantic_type", "concept_name"]]
        print(f"Created English vocab: {vocab.shape}")

    return vocab


def get_vocab_embedding(model, tokenizer, vocab, batch_size, max_length):
    model.eval()
    vocab_embedding = []
    for i in tqdm(range(0, len(vocab), batch_size)):
        batch = vocab[i:i + batch_size]
        inputs = tokenizer(batch, padding="max_length", max_length=max_length, truncation=True, return_tensors="pt")
        inputs = {k: v.to(DEVICE) for k, v in inputs.items()}
        with torch.no_grad():
            outputs = model(**inputs)
            embeddings = outputs.last_hidden_state[:, 0].cpu().numpy()
        vocab_embedding.extend(embeddings)

    return vocab_embedding


def split_sentence(text):
    sentences, sentence_ids = [], []
    chars = list(text)
    stack = ""
    idx = 0
    for c_idx, c in enumerate(chars):
        if c == "." and c_idx + 1 < len(chars) and chars[c_idx + 1] == " " and stack != "":
            sentences.append(stack)
            sentence_ids.append((idx - len(stack), idx))
            stack = ""
        elif c != "." or (c == "." and c_idx + 1 < len(chars) and chars[c_idx + 1] != " "):
            if c == " " and stack == "":
                ...
            else:
                stack += c
        idx += 1
    if stack != "":
        sentences.append(stack)
        sentence_ids.append((idx - len(stack), idx))
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


def get_span(start_idx, end_idx, num_prefix_words, num_suffix_words, doc_id, dataset_split, mention, args):
    start_idx = int(start_idx)
    end_idx = int(end_idx)
    doc_dir = os.path.join(args.base_doc_dir, doc_id[-2:], dataset_split)
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

    # print(f"Mention: {mention}")
    # print(f"Span: {span}")
    return span


def retrieval_vocab(text: str,
                    entity_id: list,
                    entity_vocab: list,
                    vocab_embedding: list,
                    model,
                    tokenizer,
                    max_length,
                    batch_size,
                    k=10):
    inputs = tokenizer(text, padding="max_length", max_length=max_length, truncation=True, return_tensors="pt")
    inputs = {k: v.to(DEVICE) for k, v in inputs.items()}
    with torch.no_grad():
        text_embedding = model(**inputs).last_hidden_state[:, 0]

    if len(vocab_embedding) < k:
        k = len(vocab_embedding)
    indices = None
    score = None

    for i in range(0, len(vocab_embedding), batch_size):
        batch = vocab_embedding[i:i + batch_size]
        batch = torch.tensor(batch).to(DEVICE)
        similarity = F.cosine_similarity(text_embedding, batch)
        values, ids = torch.topk(similarity, k=k)
        score = torch.concatenate((score, values), dim=0) if score is not None else values
        indices = torch.concatenate((indices, ids), dim=0) if indices is not None else ids
        score, ids = torch.topk(score, k=k)
        indices = indices[ids]
    indices = indices.cpu().numpy().tolist()
    labels = [entity_id[i] for i in indices]
    candidate_entity = [entity_vocab[i] for i in indices]
    return labels, candidate_entity


def batch_retrieval_vocab(batch_text: list,
                          entity_id: list,
                          entity_vocab: list,
                          vocab_embedding: list,
                          model,
                          tokenizer,
                          max_length,
                          voacb_batch_size,
                          k=10):
    model.eval()
    inputs = tokenizer(batch_text, padding="max_length", max_length=max_length, truncation=True, return_tensors="pt")
    inputs = {k: v.to(DEVICE) for k, v in inputs.items()}
    with torch.no_grad():
        text_embedding = model(**inputs).last_hidden_state[:, 0].detach().clone()

    if len(vocab_embedding) < k:
        k = len(vocab_embedding)
    indices = None
    score = None

    for i in range(0, len(vocab_embedding), voacb_batch_size):
        batch = vocab_embedding[i:i + voacb_batch_size]
        batch = torch.tensor(batch).to(DEVICE)
        similarity = F.cosine_similarity(text_embedding.unsqueeze(1), batch.unsqueeze(0), dim=-1)
        values, ids = torch.topk(similarity, k=k)
        ids += i
        score = torch.concatenate((score, values), dim=1) if score is not None else values
        indices = torch.concatenate((indices, ids), dim=1) if indices is not None else ids
        score, ids = torch.topk(score, k=k)
        indices = indices[torch.arange(ids.shape[0]).unsqueeze(1), ids]

    indices = indices.cpu().numpy().tolist()
    candidate_ids = [[entity_id[i] for i in index] for index in indices]
    candidate_entities = [[entity_vocab[i] for i in index] for index in indices]
    return candidate_ids, candidate_entities


def main(args):
    num_prefix_words = args.num_prefix_words
    num_suffix_words = args.num_suffix_words
    model_name_or_path = args.model_name_or_path
    tokenizer_name_or_path = args.tokenizer_name_or_path
    vocab_embedding_batch_size = args.vocab_embedding_batch_size
    lang = args.lang
    dataset_path = args.dataset_path
    dataset_name = args.dataset_name
    dataset_split = args.dataset_split
    max_length = args.max_length
    retrieval_batch_size = args.retrieval_batch_size

    # Load the tokenizer
    model = AutoModel.from_pretrained(model_name_or_path).to(DEVICE)
    model.eval()
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name_or_path)

    # Load the dataset

    vocab = load_dataset("andorei/BioNNE-L", "Vocabulary", split="train")
    vocab = vocab.to_pandas()
    if lang in ["en", "ru"]:
        vocab = filter_vocab(vocab, lang)

    if args.dataset_pickle_path:
        en_data_train = pd.read_pickle(args.dataset_pickle_path)
    else:
        en_data_train = load_dataset(dataset_path, dataset_name, split=dataset_split)
        en_data_train = en_data_train.to_pandas()
        if lang not in ["en", "ru"]:
            if dataset_name == "English":
                ru_data_train = load_dataset(dataset_path, "Russian", split=dataset_split)
            else:
                ru_data_train = load_dataset(dataset_path, "English", split=dataset_split)
            ru_data_train = ru_data_train.to_pandas()
            en_data_train = pd.concat([en_data_train, ru_data_train], axis=0)

    train_data = defaultdict(list)
    num_find_span4mention_error = 0

    for chem_type in ("DISO", "CHEM", "ANATOMY"):
        sub_en_data_train = en_data_train[en_data_train["entity_type"] == chem_type]
        sub_vocab = vocab[vocab["semantic_type"] == chem_type]

        entity_vocab = sub_vocab["concept_name"].to_list()
        entity_id = sub_vocab["CUI"].to_list()

        vacab_embedding = get_vocab_embedding(model, tokenizer, entity_vocab, vocab_embedding_batch_size, max_length)

        # doc_dir = os.path.join(DOC_DIR, lang, dataset_split)

        # process train data

        doc_ids = sub_en_data_train["document_id"].to_list()
        texts = sub_en_data_train["text"].to_list()
        spans = sub_en_data_train["spans"].to_list()
        CUIs = sub_en_data_train["UMLS_CUI"].to_list()
        for i in tqdm(range(0, len(doc_ids), retrieval_batch_size)):
            batch_doc_ids = doc_ids[i:i + retrieval_batch_size]
            batch_texts = texts[i:i + retrieval_batch_size]
            batch_spans = spans[i:i + retrieval_batch_size]
            batch_CUIs = CUIs[i:i + retrieval_batch_size]
            batch_span4train = []
            for j in range(len(batch_doc_ids)):
                start_idx, end_idx = batch_spans[j].split("-")[0], batch_spans[j].split("-")[-1]
                try:
                    span_include_mention = get_span(start_idx, end_idx, num_prefix_words, num_suffix_words,
                                                    batch_doc_ids[j], dataset_split, batch_texts[j], args)
                except Exception as e:
                    num_find_span4mention_error += 1
                    span_include_mention = batch_texts[j]
                batch_span4train.append(span_include_mention)
            candidate_ids, candidate_entities = batch_retrieval_vocab(batch_texts, entity_id, entity_vocab,
                                                                      vacab_embedding, model, tokenizer, max_length,
                                                                      vocab_embedding_batch_size)
            train_data["doc_id"].extend(batch_doc_ids)
            train_data["spans"].extend(batch_spans)
            train_data["mention"].extend(batch_texts)
            train_data["text"].extend(batch_span4train)
            train_data["candidates"].extend(candidate_entities)
            train_data["CUI"].extend(batch_CUIs)
            train_data["candidate_CUI"].extend(candidate_ids)
            for j in range(len(batch_CUIs)):
                labels = [1 if k == batch_CUIs[j] else 0 for k in candidate_ids[j]]
                train_data["labels"].append(labels)

    train_data = pd.DataFrame(train_data)
    train_data.to_pickle(os.path.join(DATA_DIR, args.lang, args.save_file_name))
    print(f"Number of errors in finding span4mention: {num_find_span4mention_error}")
    print(f"Saved data shape: {train_data.shape}")
    print(f"Saved data path: {os.path.join(DATA_DIR, args.lang, args.save_file_name)}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_prefix_words", type=int, default=0, help="")
    parser.add_argument("--num_suffix_words", type=int, default=0, help="")
    parser.add_argument("--model_name_or_path", type=str, default="/media/f/lichunyu/models/gebert_eng_gat", help="")
    parser.add_argument("--tokenizer_name_or_path",
                        type=str,
                        default="/media/f/lichunyu/models/gebert_eng_gat",
                        help="")
    parser.add_argument("--dataset_path", type=str, default="andorei/BioNNE-L", help="")
    parser.add_argument("--dataset_name", type=str, default="English", help="")
    parser.add_argument("--dataset_split", type=str, default="train", help="")
    parser.add_argument("--dataset_pickle_path", type=str, default="", help="")
    parser.add_argument("--base_doc_dir",
                        type=str,
                        default="/media/f/lichunyu/BioNNE-L/data/NEREL-BIO/BioNNE-L_Shared_Task/data/texts",
                        help="")
    parser.add_argument("--lang", type=str, default="en", help="")
    parser.add_argument("--vocab_embedding_batch_size", type=int, default=800, help="")
    parser.add_argument("--retrieval_batch_size", type=int, default=128, help="")
    parser.add_argument("--max_length", type=int, default=32, help="")
    parser.add_argument("--save_file_name", type=str, default="train_data.pkl", help="")
    args = parser.parse_args()
    main(args)
