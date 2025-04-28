#!/bin/bash

cd ../src
CUDA_VISIBLE_DEVICES=0 python inference.py \
--checkpoints_path /media/f/lichunyu/BioNNE-L/data/checkpoints/SapBERT-UMLS-2020AB-all-lang-from-XLMR-large-special_epoch_1_acc_0.5338.bin \
--model_name_or_path /media/f/lichunyu/models/SapBERT-UMLS-2020AB-all-lang-from-XLMR-large-special \
--tokenizer_name_or_path /media/f/lichunyu/models/SapBERT-UMLS-2020AB-all-lang-from-XLMR-large-special \
--retrieval_data_path /media/f/lichunyu/BioNNE-L/data/eeyore/ru/test_data_sap_50.pkl \
--lang ru \
--save_file_name test_ru_predictions.tsv \
--save_dir /media/f/lichunyu/BioNNE-L/data/eeyore/predictions