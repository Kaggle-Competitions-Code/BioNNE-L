#!/bin/bash


cd ../src
CUDA_VISIBLE_DEVICES=1 python preprocess_data.py \
--num_prefix_words 50 \
--num_suffix_words 50 \
--lang en \
--base_doc_dir /media/f/lichunyu/BioNNE-L/data/extra/texts \
--dataset_pickle_path /media/f/lichunyu/BioNNE-L/data/extra/pickle/en/extra_en_data.pkl \
--save_file_name train_data_extra_sap_50.pkl \
--model_name_or_path /media/f/lichunyu/models/SapBERT-from-PubMedBERT-fulltext \
--tokenizer_name_or_path /media/f/lichunyu/models/SapBERT-from-PubMedBERT-fulltext