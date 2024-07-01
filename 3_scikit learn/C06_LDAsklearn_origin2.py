#!/usr/bin/env python
# coding: utf-8

# # sklearn-LDA

# 代码示例：https://mp.weixin.qq.com/s/hMcJtB3Lss1NBalXRTGZlQ （玉树芝兰） <br>
# 可视化：https://blog.csdn.net/qq_39496504/article/details/107125284  <br>
# sklearn lda参数解读:https://scikit-learn.org/stable/modules/generated/sklearn.decomposition.LatentDirichletAllocation.html
# <br>中文版参数解读：https://blog.csdn.net/TiffanyRabbit/article/details/76445909
# <br>LDA原理-视频版：https://www.bilibili.com/video/BV1t54y127U8
# <br>LDA原理-文字版：https://www.jianshu.com/p/5c510694c07e
# <br>score的计算方法：https://github.com/scikit-learn/scikit-learn/blob/844b4be24d20fc42cc13b957374c718956a0db39/sklearn/decomposition/_lda.py#L729
# <br>主题困惑度1：https://blog.csdn.net/weixin_43343486/article/details/109255165
# <br>主题困惑度2：https://blog.csdn.net/weixin_39676021/article/details/112187210

# In[0]:
# 参数设置
processed_language = 'processed_chinese' # or eglish/chinese/all
max_iter_one = 10 # 定义单次最大迭代次数
n_top_words = 15 # 输出的词汇数量
n_topics = 20 # 定义主题的数量
max_iter = 10 # 最大迭代次数
n_max_topics = 105 # 迭代时最大主题数量
length = 5 # 每次迭代跳过的主题数量

# In[1]:

import datetime
import os
import time
import pandas as pd
import numpy as np
import re
import jieba
import jieba.analyse
import jieba.posseg as psg
from gensim import corpora, models
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import nltk
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation
import pyLDAvis
# import pyLDAvis.sklearn
import pyLDAvis.lda_model
from wordcloud import WordCloud
import matplotlib.pyplot as plt
import pandas as pd
import csv

# In[4]:


output_path = '/home/summer/PythonProject/pythonProject5/3_scikit learn/C06_lda/result2'
os.chdir(output_path)


# 可视化
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
    pyLDAvis.save_html(panel, 'lda_pass'+str(n_topics)+'_en.html')


def cost_time(func):
    def fun(*args, **kwargs):
        t = time.perf_counter()
        result = func(*args, **kwargs)
        print(f'func {func.__name__} cost time:{time.perf_counter() - t:.8f} s')
        return result

    return fun

def get_topic_word_distribution(lda_model, tf_vectorizer, top_n=15):
    """
    获取每个主题的词分布，并返回一个字典，其中键为主题编号，值为词语-权重字典。

    :param lda_model: LDA模型实例
    :param tf_vectorizer: 特征向量器实例
    :param top_n: 每个主题展示的顶级词汇数
    :return: 主题词语-权重字典
    """
    topic_word = lda_model.components_
    feature_names = tf_vectorizer.get_feature_names_out()
    topic_word_dict = {}
    for topic_idx, topic in enumerate(topic_word):
        word_weight_dict = {}
        for word_idx in topic.argsort()[:-n_top_words - 1:-1]:
            word = feature_names[word_idx]
            weight = topic[word_idx]
            word_weight_dict[word] = weight
        topic_word_dict[f"主题{topic_idx}"] = word_weight_dict
    return topic_word_dict

def generate_and_save_wordclouds(topic_word_dict, wc_width=800, wc_height=500, font_path='/mnt/c/Windows/Fonts/宋体.ttf'):
    """
    根据给定的主题词语-权重字典生成词云，并保存为图片及CSV文件。

    :param topic_word_dict: 主题词语-权重字典
    :param wc_width: 词云宽度
    :param wc_height: 词云高度
    :param font_path: 字体路径，用于支持中文显示
    """
    for topic, word_weight_dict in topic_word_dict.items():
        words_str = " ".join([f"{word} ({weight})" for word, weight in word_weight_dict.items()])
        wc = WordCloud(width=wc_width, height=wc_height, font_path=font_path, background_color='white')
        wc.generate(words_str)

        # 保存词云图
        plt.figure(figsize=(10, 10), facecolor=None)
        plt.imshow(wc)
        plt.axis("off")
        plt.tight_layout(pad=0)
        plt.savefig(f"词云_{topic}.png", dpi=300, bbox_inches='tight', pad_inches=0)
        plt.close()



def save_topic_word_dicts(topic_word_dict):
    """
    将主题词字典保存为CSV文件。

    参数:
    topic_word_dict (dict): 包含主题词的字典，其中外部键是主题编号，内部值是另一个字典，包含词和其对应的权重。

    """
    # 将主要主题词字典保存为CSV
    df = pd.DataFrame(topic_word_dict)
    df.to_csv("lda_main_topic_word_dict.csv", index=True)

    # 追加写入内部主题词到CSV
    with open("lda_inner_topic_word.csv", 'a', encoding='utf-8') as csvfile:
        writer = csv.writer(csvfile)
        
        for key, inner_dict in topic_word_dict.items():
            # 将内部字典的items展平为列表以写入CSV
            row_data = [value for item in inner_dict.items() for value in item]  # 假设inner_dict的值可直接展平
            writer.writerow(row_data)  # 写入一行数据

    print("数据已成功追加写入 lda_inner_topic_word.csv")

# In[2]:
# reading the perprocessing data
# 定义保存文件的路径
save_path = '/home/summer/PythonProject/pythonProject5/3_scikit learn/C06_lda/data2/'

# 初始化列表，用于存储处理后的英文文本、中文文本和所有文本
# 从第二次运行起，直接获取预处理过的docLst，前面load数据、预处理均注释掉
processed_english = []
processed_chinese = []
processed_all = []

# 加载处理后的英文文本
with open(save_path + 'processed_english.txt', 'r', encoding='utf-8') as f:
    # 遍历文件的每一行，去除空白字符后添加到processed_english列表中
    for line in f.readlines():
        if line != '':
            processed_english.append(line.strip())

# 加载处理后的中文文本
with open(save_path + 'processed_chinese.txt', 'r', encoding='utf-8') as f:
    # 遍历文件的每一行，去除空白字符后添加到processed_chinese列表中
    for line in f.readlines():
        if line != '':
            processed_chinese.append(line.strip())

# 加载处理后的所有文本
with open(save_path + 'processed_all.txt', 'r', encoding='utf-8') as f:
    # 遍历文件的每一行，去除空白字符后添加到processed_all列表中
    for line in f.readlines():
        if line != '':
            processed_all.append(line.strip())


# In[3]:

# ## 2.LDA分析

n_features = 1000 #提取1000个特征词语
# 2.2 创建TF向量化器
# 参数说明：
# - strip_accents: 移除字符的重音，'unicode'表示使用Unicode方式移除
# - max_features: 最多保留的特征数
# - stop_words: 停用词，'english'表示使用英文停用词
# - max_df: 最大文档频率，如果一个词在超过max_df比例的文档中出现，则被忽略
# - min_df: 最小文档频率，如果一个词在少于min_df比例的文档中出现，则被忽略
t = time.perf_counter()
tf_vectorizer = CountVectorizer(strip_accents = 'unicode',
                                max_features=n_features,
                                stop_words='english',
                                max_df = 0.5,
                                min_df = 10)
# 使用向量化器对处理后的数据进行转换
# 参数data.content_cutted为预处理后的内容
if processed_language in "processed_english":
    tf = tf_vectorizer.fit_transform(processed_english)
elif processed_language in "processed_chinese":
    tf = tf_vectorizer.fit_transform(processed_chinese)
elif processed_language in "processed_all":
    tf = tf_vectorizer.fit_transform(processed_all)
else:
    print("Error: processed_language is not in ['processed_english', 'processed_chinese', 'processed_all']")

print(f'coast time:{time.perf_counter() - t:.8f} s')

# 2.3 进行lda建模
t = time.perf_counter()
n_topics = n_topics # 定义主题的数量
lda = LatentDirichletAllocation(n_components=n_topics, max_iter=max_iter_one,
                                learning_method='batch',
                                learning_offset=50,
                                # doc_topic_prior=1/n_topics, # 这个就是arpha 值
                                # topic_word_prior=0.01, # 这个就是beta值
                                random_state=42)
lda.fit(tf) # 这一步用了十分钟
print(f'coast time:{time.perf_counter() - t:.8f} s')
print("Model building success!")

# In[3]:
# 结果分析
# ### 3.1输出每个主题对应词语

# 3.1.1 词云
# 获取每个主题的词分布

n_top_words = n_top_words # 输出的词汇数量
topic_word_dict = get_topic_word_distribution(lda, tf_vectorizer, top_n = n_top_words) # 得到主题词及其概率字典
save_topic_word_dicts(topic_word_dict)


# save_topic_word_dicts(topic_word_dict)
# df = pd.DataFrame(topic_word_dict)
# df.to_csv("lda_main_topic_word_dict.csv", index=True)

# # 假设topic_word_dict是你的字典
# with open("lda_inner_topic_word.csv", 'a', encoding='utf-8') as csvfile:  # 使用'a'模式追加写入
#     writer = csv.writer(csvfile)
    
#     for key, inner_dict in topic_word_dict.items():
#         # 将字典的items转化为列表，以便写入CSV
#         row_data = [value for item in inner_dict.items() for value in item]  # 这里假设inner_dict的值可以直接展平为一列
#         writer.writerow(row_data)  # 写入一行数据

# print("数据已成功追加写入 lda_inner_topic_word.csv")


generate_and_save_wordclouds(topic_word_dict)


# In[4]:


# ### 3.2 结果可视化
show_lda(lda,tf,tf_vectorizer,n_topics)

# In[5]:
# ### 3.3困惑度_批量计算
import matplotlib.pyplot as plt

plexs = []
scores = []
n_max_topics = n_max_topics # 迭代时最大主题数量
length = length # 每次迭代跳过的主题数量
start_value = 1

for i in range(1,n_max_topics, length):
    try:
        time.perf_counter()
        print(i)
        lda = LatentDirichletAllocation(n_components=i, max_iter=max_iter,
                                        learning_method='batch',
                                        # doc_topic_prior=1 / i,  # 这个就是arpha 值
                                        # topic_word_prior=0.01,  # 这个就是beta值
                                        learning_offset=50,random_state=42)
        lda.fit(tf)
        plexs.append(lda.perplexity(tf))
        scores.append(lda.score(tf))
        if i < 5:
            start_value = i+1
            continue
        else:
            show_lda(lda, tf, tf_vectorizer, i)
            topic_word_dict = get_topic_word_distribution(lda, tf_vectorizer, n_top_words)
            # save_topic_words_to_csv(lda, tf_vectorizer, i, n_top_words)
            df = pd.DataFrame(topic_word_dict)
            df.to_csv(f"lda_{i}topic_word_dict.csv", index=True)
        print(f'coast time:{time.perf_counter() - t:.8f} s')
    except:
        continue


# wirte perplexity in to csv
list_length = len(plexs)
interval = length
end_value = start_value+(list_length)*interval
x = np.arange(start_value, end_value, interval)
df = pd.DataFrame({"x":x,"plexs": plexs})
df.to_csv('lda_perplexity.csv',index=False)

plt.figure(figsize=(30,30))
plt.plot(x,plexs)
plt.xlabel("number of topics")
plt.ylabel("perplexity")
plt.show()
plt.savefig("lda_perplexity.png", dpi=300, bbox_inches='tight', pad_inches=0)
