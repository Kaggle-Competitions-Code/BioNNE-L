# -*- encoding: utf-8 -*-
'''
@create_time: 2025/03/21 20:47:10
@author: lichunyu
'''

from relik import Relik
from relik.inference.data.objects import RelikOutput
from transformers import AutoModel, AutoTokenizer

relik = Relik.from_pretrained("sapienzanlp/relik-entity-linking-large")
relik_out: RelikOutput = relik("Michael Jordan was one of the best players in the NBA.")
...