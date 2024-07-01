"""
使用gensim实现LDA
请在LDA21环境中使用此程序
"""
#!/usr/bin/env python
# coding: utf-8

import gensim
import os
import matplotlib.pyplot as plt
import time
import pyLDAvis
import pyLDAvis.gensim_models
import pandas as pd
import csv
import numpy as np
from gensim import corpora, models
from wordcloud import WordCloud


# 一次生成
num_topics = 20 # 主题数
n_topics_word = 20 # 展示前N个主题词
passes = 10 # 最大迭代次数
processed_language = 'processed_chinese' # or english/chinese/all

# 多次迭代
start_value = 17 # 开始迭代数
n_max_topics = 21 # 最大主题数量
length = 1 # 每次迭代跳过的主题数量
max_iter = 10 # 最大迭代次数

# 数据路径
save_path = "/home/summer/PythonProject/pythonProject5/8_genmis/C01_LDA/result2/"
os.chdir(save_path)
save_path = '/home/summer/PythonProject/pythonProject5/8_genmis/C01_LDA/data2/'



def get_lda_model(num_topics, corpus, id2word, texts, passes):
    # set training parameters
    chunksize = 200  # 一次处理的文档数量
    iterations = 400  # 在每个主题-词分布上运行和优化LDA模型迭代的次数
    eval_every = None


    # Initialize the LDA model
    lda_model = gensim.models.ldamodel.LdaModel(
        corpus=corpus,
        id2word=id2word,
        num_topics=num_topics,
        passes=passes,
        alpha='auto',
        eta='auto',
        eval_every=eval_every,
        chunksize=chunksize,
        iterations=iterations
    )

    # Compute perplexity
    perplexity = lda_model.log_perplexity(corpus)

    # Correctly reference id2word instead of an undefined variable 'dictionary'
    coherence_model_lda = gensim.models.CoherenceModel(
        model=lda_model,
        texts=texts,
        dictionary=id2word,  # This was likely the source of error
        coherence='c_v'
    )

    # Now compute coherence
    coherence_lda = coherence_model_lda.get_coherence()


    return lda_model, perplexity, coherence_lda


def save_lda_visualization(lda_model, corpus, dictionary, num_topics):
    """
    Prepare and save an LDA visualization using pyLDAvis.

    Parameters:
    - lda_model: Trained LDA model from gensim.
    - corpus: The corpus in gensim's bag-of-words format.
    - dictionary: The mapping of word IDs to words.
    - num_topics: The number of topics used in the LDA model.

    Returns:
    None
    """
    # Prepare the visualization data
    vis_data = pyLDAvis.gensim_models.prepare(lda_model, corpus, dictionary)

    # Save the visualization as an HTML file
    filename = f'lda_pass_{num_topics}.html'
    pyLDAvis.save_html(vis_data, filename)


def get_topic_word_distribution_gensim(lda_model, id2word, top_n=15, num_topics=20):
    """
    获取每个主题的词分布，并返回一个字典，其中键为主题编号，值为词语-权重字典。

    :param lda_model: Gensim LDA模型实例
    :param id2word: 字典，映射词语ID到词语
    :param top_n: 每个主题展示的顶级词汇数
    :return: 主题词语-权重字典
    """
    topic_word_dict = {}
    for topic_idx, topic_list in enumerate(lda_model.show_topics(num_words=top_n, formatted=False, num_topics=num_topics)):
        topic_word_dict[f"主题{topic_idx}"] = {word: weight for word, weight in topic_list[1]}
    return topic_word_dict


def save_topic_word_dicts(topic_word_dict, num_topics):
    """
    将主题词字典保存为CSV文件。

    参数:
    topic_word_dict (dict): 包含主题词的字典，其中外部键是主题编号，内部值是另一个字典，包含词和其对应的权重。

    """
    # 将主要主题词字典保存为CSV
    df = pd.DataFrame(topic_word_dict)
    df.to_csv(f"lda_{num_topics}topic_word_dict.csv", index=True)

    # 追加写入内部主题词到CSV
    with open(f"lda_{num_topics}inner_topic_word.csv", 'a', encoding='utf-8') as csvfile:
        writer = csv.writer(csvfile)

        for key, inner_dict in topic_word_dict.items():
            # 将内部字典的items展平为列表以写入CSV
            row_data = [value for item in inner_dict.items() for value in item]  # 假设inner_dict的值可直接展平
            writer.writerow(row_data)  # 写入一行数据

    print("数据已成功追加写入 lda_inner_topic_word.csv")


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


def cost_time(func):
    def fun(*args, **kwargs):
        t = time.perf_counter()
        result = func(*args, **kwargs)
        print(f'func {func.__name__} cost time:{time.perf_counter() - t:.8f} s')
        return result

    return fun
# In[2]:

# reading the perprocessing data
def load_processed_docs(language, save_path):
    """Load preprocessed documents based on the specified language."""
    file_name = f"processed_{language}.txt"
    file_path = save_path + file_name

    processed_docs = []
    current_doc = []

    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            if line == "":
                pass
            else:
                current_doc.extend(line.split())
                processed_docs.append(current_doc)
                current_doc = []

    return processed_docs

# Main code
processed_content = []
processed_language = "english"  # Replace with actual variable or string

if processed_language in ["english", "chinese", "all"]:
    processed_content = load_processed_docs(processed_language, save_path)
else:
    print("language error")

print([doc for doc in processed_content])
print([type(doc) for doc in processed_content])
