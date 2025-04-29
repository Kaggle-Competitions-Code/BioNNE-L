#!/bin/bash

cd ../src
CUDA_VISIBLE_DEVICES=0 python step2_train.py \
--train_data_path /media/f/lichunyu/BioNNE-L/data/eeyore/en/train_data_with_extra_sap_50.pkl \
--retrieval_topk 5 \
--train_batch_size 8 \
--eval_batch_size 8 \
--learning_rate 1e-5 \
--num_train_epochs 5 \
--model_name_or_path /media/f/lichunyu/models/BiomedNLP-BiomedBERT-base-uncased-abstract-special \
--tokenizer_name_or_path /media/f/lichunyu/models/BiomedNLP-BiomedBERT-base-uncased-abstract-special \
--do_eval