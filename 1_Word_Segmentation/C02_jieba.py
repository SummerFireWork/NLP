"""
学习如何使用jieba库进行中文分词。
"""

import jieba
import jieba.posseg as psg
import re
import pandas as pd

def get_stop_dict(file):
    content = open(file, encoding='utf-8')
    word_list = []
    for c in content:
        c = re.sub('\n|\r', '', c)