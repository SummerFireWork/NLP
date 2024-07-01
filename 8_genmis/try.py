
import os
import re
import jieba
import jieba.analyse
import jieba.posseg as psg
from gensim import corpora, models
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import nltk
import string
from nltk.stem.porter import PorterStemmer
from os import path


# get data directory (using getcwd() is needed to support running example in generated IPython notebook)
home_path = os.getcwd()
# Read the whole text.
output_path = path.join(home_path, 'C01_LDA')
dic_file = path.join(home_path, 'C01_LDA/stop_dic/dict.txt')
stop_file = path.join(home_path, 'C01_LDA/stop_dic/stopwords.txt')
file_path = path.join(home_path, 'C01_LDA/data2/data.txt')
save_path = path.join(home_path, 'C01_LDA/data2/')

stop_list = []
try:
    with open(stop_file, encoding='utf-8') as stopword_list:
        for line in stopword_list:
            line = line.strip()
            stop_list.append(line)
        stoplist = list(set(stop_list).union(set(stopwords.words('english'))))
except Exception as e:
    print(f"Error loading stop words: {e}")
    stop_list = []


