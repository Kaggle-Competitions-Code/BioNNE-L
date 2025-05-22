#!/bin/bash


cd ../src
CUDA_VISIBLE_DEVICES=1 python preprocess_data.py \
--num_prefix_words 50 \
--num_suffix_words 50 \
--dataset_split dev \
--save_file_name dev_data_sap_50.pkl \
--model_name_or_path /media/f/lichunyu/models/SapBERT-from-PubMedBERT-fulltext \
--tokenizer_name_or_path /media/f/lichunyu/models/SapBERT-from-PubMedBERT-fulltext