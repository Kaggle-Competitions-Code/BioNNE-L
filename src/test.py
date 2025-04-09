# -*- encoding: utf-8 -*-
'''
@create_time: 2025/04/09 15:05:10
@author: lichunyu
'''
import pandas as pd
from datasets import load_dataset

dataset = load_dataset("adlbh/umls-concepts", split="train")
umls = dataset.to_pandas()

df = pd.read_pickle("/media/f/lichunyu/BioNNE-L/data/eeyore/dev_data.pkl")
...
