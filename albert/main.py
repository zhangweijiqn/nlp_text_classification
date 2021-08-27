import torch
from transformers import BertTokenizer,BertModel,BertConfig
import numpy as np
from torch.utils import data
from sklearn.model_selection import train_test_split
import pandas as pd

pretrained = 'voidful/albert_chinese_small' #使用small版本Albert
tokenizer = BertTokenizer.from_pretrained(pretrained)
model=BertModel.from_pretrained(pretrained)
config=BertConfig.from_pretrained(pretrained)

inputtext = "今天心情情很好啊，买了很多东西，我特别喜欢，终于有了自己喜欢的电子产品，这次总算可以好好学习了"
tokenized_text=tokenizer.encode(inputtext)
input_ids=torch.tensor(tokenized_text).view(-1,len(tokenized_text))
outputs=model(input_ids)

