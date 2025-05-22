#!/bin/bash


cd ../src
CUDA_VISIBLE_DEVICES=0 python preprocess_data.py \
--num_prefix_words 50 \
--num_suffix_words 50 \
--dataset_split train \
--save_file_name train_data_sap_50.pkl \
--model_name_or_path /media/f/lichunyu/models/SapBERT-UMLS-2020AB-all-lang-from-XLMR-large \
--tokenizer_name_or_path /media/f/lichunyu/models/SapBERT-UMLS-2020AB-all-lang-from-XLMR-large \
--lang bilingual \
--vocab_embedding_batch_size 600