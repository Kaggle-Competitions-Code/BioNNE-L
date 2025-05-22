#!/bin/bash

cd ../src
CUDA_VISIBLE_DEVICES=1 python inference.py \
--checkpoints_path /media/f/lichunyu/BioNNE-L/data/checkpoints/bilingual/SapBERT-UMLS-2020AB-all-lang-from-XLMR-large-special_epoch_0_acc_0.6576.bin \
--model_name_or_path /media/f/lichunyu/models/SapBERT-UMLS-2020AB-all-lang-from-XLMR-large-special \
--tokenizer_name_or_path /media/f/lichunyu/models/SapBERT-UMLS-2020AB-all-lang-from-XLMR-large-special \
--retrieval_data_path /media/f/lichunyu/BioNNE-L/data/eeyore/bilingual/test_data_sap_50.pkl \
--lang bilingual \
--save_file_name test_bilingual_predictions.tsv \
--save_dir /media/f/lichunyu/BioNNE-L/data/eeyore/predictions