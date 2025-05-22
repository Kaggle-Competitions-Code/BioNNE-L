#!/bin/bash

cd ../src
CUDA_VISIBLE_DEVICES=0 python step2_train.py \
--train_data_path /media/f/lichunyu/BioNNE-L/data/eeyore/bilingual/train_data_sap_50.pkl \
--dev_data_path /media/f/lichunyu/BioNNE-L/data/eeyore/bilingual/dev_data_sap_50.pkl \
--retrieval_topk 5 \
--train_batch_size 4 \
--eval_batch_size 4 \
--learning_rate 1e-5 \
--num_train_epochs 5 \
--model_name_or_path /media/f/lichunyu/models/SapBERT-UMLS-2020AB-all-lang-from-XLMR-large-special \
--tokenizer_name_or_path /media/f/lichunyu/models/SapBERT-UMLS-2020AB-all-lang-from-XLMR-large-special \
--save_dir /media/f/lichunyu/BioNNE-L/data/checkpoints/bilingual \
--do_eval