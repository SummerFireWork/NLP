"""
使用gensim实现LDA
请在LDA21环境中使用此程序
"""
#!/usr/bin/env python
# coding: utf-8

import gensim
import jieba
import jieba.posseg as psg
import re
import os
import matplotlib.pyplot as plt
import matplotlib
import nltk
import pyLDAvis
import pyLDAvis.gensim_models
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer


# 确保nltk资源已下载
nltk.download('stopwords')
nltk.download('wordnet')

dic_file = "D:\\pythonProject5\\8_genmis\\C01_LDA\\stop_dic\\dict.txt"
stop_file = "D:\\pythonProject5\\8_genmis\\C01_LDA\\stop_dic\\stopwords.txt"
# 英文预处理函数
def preprocess_english(text):
    # 转换为小写，分词，去停用词，词形还原
    text = re.sub(r'\W', ' ', text.lower())  # 去除非字母字符
    tokens = text.split()
    stop_words = set(stopwords.words('english'))
    tokens = [token for token in tokens if token not in stop_words]
    lemmatizer = WordNetLemmatizer()
    tokens = [lemmatizer.lemmatize(token) for token in tokens]
    return tokens

# 中文预处理函数，使用jieba分词
def preprocess_chinese(text):
    jieba.load_userdict(dic_file)
    jieba.initialize()
    try:
        stopword_list = open(stop_file, encoding='utf-8')
    except:
        stopword_list = []
        print("error in stop_file")
    stop_list = []
    flag_list = ['n', 'nz', 'vn']
    for line in stopword_list:
        line = re.sub(u'\n|\\r', '', line)
        stop_list.append(line)

    word_list = []
    # jieba分词
    seg_list = psg.cut(text)
    # print(list(seg_list))
    for seg_word in seg_list:
        # word = re.sub(u'[^\u4e00-\u9fa5]','',seg_word.word)
        word = seg_word.word  # 如果想要分析英语文本，注释这行代码，启动下行代码
        find = 0
        for stop_word in stop_list:
            if stop_word == word or len(word) < 2:  # this word is stopword
                find = 1
                break
        if find == 0 and seg_word.flag in flag_list:
            word_list.append(word)
    # print(word_list)  # 打印分词结果
    return word_list


# 分离中英文
def separate_cne(text):
    chinese_pattern = re.compile(r'[\u4e00-\u9fa5]+')
    english_pattern = re.compile(r'[a-zA-Z0-9]+')
    chinese_parts = chinese_pattern.findall(text)
    english_parts = english_pattern.findall(text)
    return ' '.join(chinese_parts), ' '.join(english_parts)

def show_lda(lda,tf,tf_vectorizer,n_topics):
    # 计算文档长度和词频
    doc_length = tf.sum(axis=1).getA1()  # 获取每个文档的总词数
    term_frequency = tf.sum(axis=0).getA1()  # 获取每个词的全局词频

    panel = pyLDAvis.prepare(
        topic_term_dists=lda.components_,
        doc_topic_dists=lda.transform(tf),
        doc_lengths=doc_length,
        vocab=tf_vectorizer.get_feature_names_out(),
        term_frequency=term_frequency

    )
    pyLDAvis.display(panel)
    pyLDAvis.save_html(panel, 'lda_pass'+str(n_topics)+'.html')

def get_lda_model(num_topics, corpus, id2word, text):
    # set training parameters
    num_topics = num_topics
    chunksize = 200 # 一次处理的文档数量
    passes = 20 # 在语料库上运行和优化LDA模型迭代的次数
    iterations = 400 # 在每个主题-词分布上运行和优化LDA模型迭代的次数
    eval_every = None

    x = []  # x轴
    perplexity_values = []  # 困惑度
    coherence_values = []  # 一致性 现在运算失败，无法计算一致性

    for i in range(2, num_topics + 1):
        lda_model = gensim.models.ldamodel.LdaModel(corpus=corpus, id2word=id2word, num_topics=i, passes=passes,
                                                    alpha='auto', eta='auto', eval_every=eval_every,
                                                    chunksize=chunksize, iterations=iterations)
        # lda_model = gensim.models.ldamodel.LdaModel(corpus=corpus, id2word=dictionary, num_topics=i, passes=50)
        perplexity = lda_model.log_perplexity(corpus)
        coherence_model_lda = gensim.models.CoherenceModel(model=lda_model, texts=text, dictionary=dictionary,
                                                           coherence='c_v')
        # coherence_lda = coherence_model_lda.get_coherence()

        x.append(i)
        perplexity_values.append(perplexity)
        # coherence_values.a5nd(coherence_lda)
        # show_lda(lda_model,corpus,dictionary,i)
        vis_data = pyLDAvis.gensim_models.prepare(lda_model, corpus, dictionary)
        # pyLDAvis.show(vis_data)
        # pyLDAvis.display(vis_data)
        pyLDAvis.save_html(vis_data, 'lda_pass'+str(i)+'.html')

    return x, perplexity_values, coherence_values


##########################################
# 导入文本

with open('D:\\pythonProject5\\8_genmis\\C01_LDA\\data\\data.txt', 'r', encoding='utf-8') as f:
    doc = f.read()
    f.close()

# 分离中英文
chinese_text, english_text = separate_cne(doc)

# 分词
processed_chinese = preprocess_chinese(chinese_text)
processed_english = preprocess_english(english_text)
all_tokens = processed_chinese + processed_english

# LDA建模
import gensim.corpora as corpora

dictionary = gensim.corpora.Dictionary([processed_english])
dictionary.filter_extremes(no_below=10, no_above=0.5) # 删除出现少于20个文档的单词或在50％以上文档中出现的单词
corpus = [dictionary.doc2bow(tmp) for tmp in [processed_english]]

# Make a index to word dictionary
temp = dictionary[0]  # 0th item in the dictionary is the first word
id2word = dictionary.token2id  # 得到词和id的对应关系
dictionary.save('D:\\pythonProject5\\8_genmis\\C01_LDA\\data\\dictionary.gensim')
dictionary.save_as_text('D:\\pythonProject5\\8_genmis\\C01_LDA\\data\\dictionary.txt')
dictionary = gensim.corpora.Dictionary.load('D:\\pythonProject5\\8_genmis\\C01_LDA\\data\\dictionary.gensim')
dictionary = gensim.corpora.Dictionary.load_from_text('D:\\pythonProject5\\8_genmis\\C01_LDA\\data\\dictionary.txt')

# 计算困惑度和coherence
num_topics = 100
x, perplexity_values, coherence_values = get_lda_model(num_topics, corpus, id2word, processed_english)
print(x)
print(perplexity_values)