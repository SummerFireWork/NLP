#!/usr/bin/env python
# coding: utf-8

import gensim
import jieba
import re
import os
import matplotlib.pyplot as plt
import matplotlib
import pyLDAvis
import pyLDAvis.gensim_models

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

def get_lda_model(num_topics, corpus, dictionary, text):
    x = []  # x轴
    perplexity_values = []  # 困惑度
    coherence_values = []  # 一致性 现在运算失败，无法计算一致性

    for i in range(2, num_topics + 1):
        # lda_model = gensim.models.ldamodel.LdaModel(corpus=corpus, id2word=dictionary, num_topics=i, passes=10,
        #                                             chunksize=50, iterations=400)
        lda_model = gensim.models.ldamodel.LdaModel(corpus=corpus, id2word=dictionary, num_topics=i, passes=10)
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
# 导入文本和停用词
import os
os.chdir('D:\\pythonProject5\\8_genmis\\C01_LDA')

stop_words = []
with open('stop_dic/stopwords.txt', 'r', encoding='utf-8') as f:
    for line in f.readlines():
        stop_words.append(line.strip())
    f.close()

with open('data/data.txt', 'r', encoding='utf-8') as f:
    sentence = f.read()
    f.close()

#################################
# 分词并建模

import jieba

text = []

for word in jieba.lcut(sentence):
    if word not in stop_words:
        text.append(word)

import gensim.corpora as corpora

dictionary = gensim.corpora.Dictionary([text])
corpus = [dictionary.doc2bow(tmp) for tmp in [text]]

num_topics = 100
x, perplexity_values, coherence_values = get_lda_model(num_topics, corpus, dictionary, text)
print(x)
print(perplexity_values)