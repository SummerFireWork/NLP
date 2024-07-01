# -*- coding:utf-8 -*-
"""
本程序是最简单的word2vec，训练一个Word2Vec模型，并使用它来生成单词向量。
"""
# 导入必要的库
from gensim.models import word2vec
import logging

# 配置日志记录
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

# 原始句子列表
raw_sentence = ["the quick brown fox jumped over the lazy dog", "the cat jumped over the lazy dog"]

# 将原始句子转换为小写，并分割为单词列表
sentence = [word.lower().split() for word in raw_sentence]
print(sentence)

# 使用Word2Vec模型训练单词向量
model = word2vec.Word2Vec(sentence, min_count=1)
