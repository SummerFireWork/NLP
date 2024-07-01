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
from os import path
import logging
import math

# 一次生成
num_topics = 5 # 主题数
n_topics_word = 20 # 展示前N个主题词
passes = 10 # 最大迭代次数
processed_language = "english"  #or english/chinese/all

# 多次迭代
start_value = 5 # 开始迭代数
n_max_topics = 50 # 最大主题数量
length = 1 # 每次迭代跳过的主题数量
max_iter = 10 # 最大迭代次数


# get data directory (using getcwd() is needed to support running example in generated IPython notebook)
home_path = os.getcwd()
# Read the whole text.
result_path = path.join(home_path, 'C01_LDA/result/')
os.chdir(result_path)
save_path = path.join(home_path, 'C01_LDA/data/')


# 建立日志
# 创建一个logger
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)  # 设置日志级别，这里是DEBUG
# 创建一个handler，用于写入日志文件
fh = logging.FileHandler('LDA.log')
fh.setLevel(logging.DEBUG)  # handler的级别也要设置
# 定义handler的输出格式
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
fh.setFormatter(formatter)
# 给logger添加handler
logger.addHandler(fh)


def get_lda_model(num_topics, corpus, id2word, texts, passes):
    # set training parameters
    chunksize = 200  # 一次处理的文档数量 让计算机自动处理好了
    iterations = 400  # 在每个主题-词分布上运行和优化LDA模型迭代的次数 让计算机自动处理好了
    eval_every = None

    lda_model = gensim.models.LdaMulticore(
        corpus=corpus,
        id2word=id2word,
        num_topics=num_topics,
        passes=passes,
        # alpha='auto',
        # eta='auto',
        eval_every=eval_every,
        # chunksize=chunksize,
        # iterations=iterations,
        random_state=42,
       #  per_word_topics=True,
       #  minimum_probability=0.0,
       #  minimum_phi_value=0.0,
       #  decay=0.5,
       #  offset=1.0,
       #  eta=None,
       #  lower_bound=-1e100,
       #  upper_bound=1e100,
       #  max_memory_size=1000,
       #  dtype=np.float32,
       #  distributed=False,
       #  callbacks=None,
       # gamma_threshold=0.001,
        workers=None
    )


    # # Initialize the LDA model
    # lda_model = gensim.models.ldamodel.LdaModel(
    #     corpus=corpus,
    #     id2word=id2word,
    #     num_topics=num_topics,
    #     passes=passes,
    #     alpha='auto',
    #     eta='auto',
    #     eval_every=eval_every,
    #     chunksize=chunksize,
    #     iterations=iterations
    # )

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

def perplexity(ldamodel, testset, dictionary, size_dictionary, num_topics):
    print('the info of this ldamodel: \n')
    print('num of topics: %s' % num_topics)
    prep = 0.0
    prob_doc_sum = 0.0
    topic_word_list = []
    for topic_id in range(num_topics):
        topic_word = ldamodel.show_topic(topic_id, size_dictionary)
        dic = {}
        for word, probability in topic_word:
            dic[word] = probability
        topic_word_list.append(dic)
    doc_topics_ist = []
    for doc in testset:
        doc_topics_ist.append(ldamodel.get_document_topics(doc, minimum_probability=0))
    testset_word_num = 0
    for i in range(len(testset)):
        prob_doc = 0.0
        doc = testset[i]
        doc_word_num = 0
        for word_id, num in dict(doc).items():
            prob_word = 0.0
            doc_word_num += num
            word = dictionary[word_id]
            for topic_id in range(num_topics):
                # cal p(w) : p(w) = sumz(p(z)*p(w|z))
                prob_topic = doc_topics_ist[i][topic_id][1]
                prob_topic_word = topic_word_list[topic_id][word]
                prob_word += prob_topic * prob_topic_word
            prob_doc += math.log(prob_word)  # p(d) = sum(log(p(w)))
        prob_doc_sum += prob_doc
        testset_word_num += doc_word_num
    prep = math.exp(-prob_doc_sum / testset_word_num)  # perplexity = exp(-sum(p(d)/sum(Nd))
    print("模型困惑度为 : %s" % prep)
    return prep

def graph_draw(topic, perplexity):
    x = topic
    y = perplexity
    plt.plot(x, y, color="red", linewidth=2)
    plt.xlabel("Number of Topic")
    plt.ylabel("Perplexity")
    plt.savefig("Perplexity-Topics")
    plt.show()


# In[2]:

# Main code
processed_content = []

if processed_language in ["english", "chinese", "all"]:
    processed_content = load_processed_docs(processed_language, save_path)
else:
    print("language error")

# In[3]:
# LDA建模


dictionary = gensim.corpora.Dictionary(processed_content)
# dictionary.filter_extremes(no_below=10, no_above=0.5) # 删除出现少于20个文档的单词或在50％以上文档中出现的单词 # 加上这一行的时候就会报错
corpus = [dictionary.doc2bow(tmp) for tmp in processed_content]
corpora.MmCorpus.serialize('corpus.mm', corpus)
i = 20
# 尝试新困惑度算法
for i in range(20,21,1): # 需要考证，抽样的含义是什么。
    print("抽样为"+str(i)+"时的perplexity")
    lda_model, perplexity_value, coherence_value = get_lda_model(num_topics, corpus, dictionary, processed_content, passes)
    corpus = corpora.MmCorpus('corpus.mm')
    testset = []
    for c in range(int(corpus.num_docs/i)):
        testset.append(corpus[c*i])
    prep = perplexity(lda_model, testset, dictionary, len(dictionary.keys()), num_topics) # 这里生成的是自定义困惑度


print(f"主题数为{num_topics}模型建立完成, 困惑度为{perplexity_value},抽样为str{i}时的困惑度为{prep}, 一致性为{coherence_value}, 最大迭代次数为{passes}")
logger.info(f"主题数为{num_topics}模型建立完成, 困惑度为{perplexity_value},抽样为str{i}时的困惑度为{prep}, 一致性为{coherence_value}, 最大迭代次数为{passes}")


# In[4]:
# 可视化

# Example usage:
# Assuming lda_model, corpus, and dictionary are already defined
save_lda_visualization(lda_model, corpus, dictionary, num_topics)

# In[5]:
# 数据读取和分析
topic_word_dict = get_topic_word_distribution_gensim(lda_model, dictionary, top_n=n_topics_word, num_topics=num_topics)
save_topic_word_dicts(topic_word_dict, num_topics)


# In[6]:

# 词云
print("正在生成词云...")
# generate_and_save_wordclouds(topic_word_dict)

# In[7]:
# 迭代
plexs = []
plexs_new = []
coherence = []
n_max_topics = n_max_topics # 迭代时最大主题数量
length = length # 每次迭代跳过的主题数量
start_value = start_value

for j in range(20, 21, 1): # 定义抽样次数
    print("抽样为"+str(i)+"时的perplexity")
    for i in range(start_value,n_max_topics, length): # i是主题个数
        # try:
        t = time.perf_counter()
        time.perf_counter()

        # 1. LDA建模
        lda_model, perplexity_value, coherence_value = get_lda_model(i, corpus, dictionary,
                                                                       processed_content, max_iter)
        # 2. 计算新困惑度
        corpus = corpora.MmCorpus('corpus.mm')
        testset = []
        for c in range(int(corpus.num_docs/j)):
            testset.append(corpus[c*j])

        prep = perplexity(lda_model, testset, dictionary, len(dictionary.keys()), i) # 这里生成的是自定义困惑度

        # 2.1 保存困惑度、一致性
        plexs_new.append(prep)
        plexs.append(perplexity_value)
        coherence.append(coherence_value)

        # 3. 保存可视化结果
        save_lda_visualization(lda_model, corpus, dictionary, i)
        # 4. 保存词分布
        topic_word_dict = get_topic_word_distribution_gensim(lda_model, dictionary, top_n=n_topics_word, num_topics=i)
        save_topic_word_dicts(topic_word_dict, i)
        # 5. wirte perplexity in to csv
        list_length = len(plexs)
        interval = length
        end_value = start_value + (list_length) * interval
        x = np.arange(start_value, end_value, interval)
        df = pd.DataFrame({"x": x, "plexs": plexs, "plexs_new":plexs_new, "conherence": coherence})
        df.to_csv('lda_perplexity.csv', index=False)
        print(f'coast time:{time.perf_counter() - t:.8f} s')
        logger.info(f'coast time:{time.perf_counter() - t:.8f} s')
        print(f"主题数量为{i}")
        logger.info(f"主题数量为{i}")
        print("模型新困惑度为"+str(prep))
    graph_draw(list(range(start_value, n_max_topics, length)), plexs_new)