#!/usr/bin/env python
# coding: utf-8

# In[ ]:


#nltk-LookupError错误：https://blog.csdn.net/yeshang_lady/article/details/105885533


# ## 小试牛刀

# In[9]:


from nltk import word_tokenize


# In[10]:


text = "It's a leading platform for building Python programs to work with human language data. It provides easy-to-use interfaces to over 50 corpora and lexical resources such as WordNet, along with a suite of text processing libraries for classification, tokenization, stemming, tagging, parsing, and semantic reasoning, wrappers for industrial-strength NLP libraries, and an active discussion forum."


# In[11]:


tokens = word_tokenize(text)


# In[12]:


print(tokens) #缩略词，标点，easy-to-use


# In[13]:


#https://blog.csdn.net/qq_28790663/article/details/116719867
from nltk.stem import WordNetLemmatizer
wnl = WordNetLemmatizer()


# In[14]:


wnl.lemmatize('saddest', 'a')


# In[15]:


#分别定义需要进行还原的单词与相对应的词性
words = ['cars','men','running','ate','saddest','fancier']
pos_tags = ['n','n','v','v','a','a']
 
for i in range(len(words)):
    print(words[i]+'--'+pos_tags[i]+'-->'+wnl.lemmatize(words[i],pos_tags[i]))


# ## 读取文本文件分析词频

# In[16]:


import re
import nltk
import string
import pandas as pd
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk import MWETokenizer
from nltk.corpus import wordnet
wnl = WordNetLemmatizer()


# In[17]:


def get_word_pos(tag):
    if tag.startswith('J'):
        return wordnet.ADJ
    elif tag.startswith('V'):
        return wordnet.VERB
    elif tag.startswith('N'):
        return wordnet.NOUN
    elif tag.startswith('R'):
        return wordnet.ADV
    else:
        return None


# In[18]:


text = open("./data.txt",encoding='utf-8').readlines()


# In[19]:


word_dict = {}
for t in text:
    #去除标点符号
    for c in string.punctuation:
        if c !='-':
            t = t.replace(c,' ')
    #分词，添加自定义词组，去除停用词
    tokenizer = MWETokenizer([('Python', 'programs'), ('a', 'little', 'bit'), ('a', 'lot')], separator = '-')
    wordlist = tokenizer.tokenize(nltk.word_tokenize(t))
    #wordlist = nltk.word_tokenize(t)
    filtered = [w for w in wordlist if w not in stopwords.words('english')]
    #标注词性
    refiltered =nltk.pos_tag(filtered)
    #词形还原
    lemmas_sent = []
    for wordtag in refiltered:
        wordnet_pos = get_word_pos(wordtag[1]) or wordnet.NOUN
        word = wnl.lemmatize(wordtag[0], pos=wordnet_pos)
        lemmas_sent.append(word) # 词形还原
        word_dict[word] = word_dict.get(word,0)+1


# In[20]:


wordfreq = pd.DataFrame({'word':word_dict.keys(),'freq':word_dict.values()})
wordfreq.to_excel("wordfreq.xlsx",index=False)




