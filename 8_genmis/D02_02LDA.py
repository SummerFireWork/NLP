"""
能算，但是参数有问题。
"""
import os
import logging
import gensim
import pyLDAvis
import pyLDAvis.gensim_models
import pandas as pd
import numpy as np
from gensim import corpora, models
from wordcloud import WordCloud
from os import path
import time
import matplotlib.pyplot as plt
import math
import csv

# 代码中使用的常量定义
N_TOPICS_WORD = 20 # 每个主题的关键词数量
PASSES = 10 # 迭代次数
PROCESSED_LANGUAGE = "english" # 预处理后的语言
START_VALUE = 5 # 初始化主题数的值
N_MAX_TOPICS = 50 # 最大主题数
LENGTH = 1 # 主题词间隔数
j = 20 # 肘点法困惑度计算，文档分隔数量

# 日志配置
# logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
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

# 工具函数和逻辑
def get_lda_model(num_topics, corpus, id2word, texts, passes):
    chunksize = 200
    iterations = 400
    eval_every = None

    lda_model = gensim.models.LdaMulticore(
        corpus=corpus,
        id2word=id2word,
        num_topics=num_topics,
        passes=passes,
        random_state=42,
    )

    perplexity = lda_model.log_perplexity(corpus)

    coherence_model_lda = gensim.models.CoherenceModel(
        model=lda_model,
        texts=texts,
        dictionary=id2word,
        coherence='c_v'
    )

    coherence_lda = coherence_model_lda.get_coherence()

    return lda_model, perplexity, coherence_lda

def save_lda_visualization(lda_model, corpus, dictionary, num_topics):
    vis_data = pyLDAvis.gensim_models.prepare(lda_model, corpus, dictionary)
    pyLDAvis.save_html(vis_data, f'lda_pass_{num_topics}.html')

def get_topic_word_distribution_gensim(lda_model, id2word, top_n=15, num_topics=20):
    topic_word_dict = {}
    for topic_idx, topic_list in enumerate(lda_model.show_topics(num_words=top_n, formatted=False, num_topics=num_topics)):
        topic_word_dict[f"主题{topic_idx}"] = {word: weight for word, weight in topic_list[1]}
    return topic_word_dict

def save_topic_word_dicts(topic_word_dict, num_topics):
    df = pd.DataFrame(topic_word_dict)
    df.to_csv(f"lda_{num_topics}topic_word_dict.csv", index=True)

    with open(f"lda_{num_topics}inner_topic_word.csv", 'a', encoding='utf-8') as csvfile:
        writer = csv.writer(csvfile)
        for inner_dict in topic_word_dict.values():
            row_data = [value for item in inner_dict.items() for value in item]
            writer.writerow(row_data)

def generate_and_save_wordclouds(topic_word_dict, num_topics, wc_width=800, wc_height=500, font_path='/mnt/c/Windows/Fonts/宋体.ttf'):
    for topic, word_weight_dict in topic_word_dict.items():
        words_str = " ".join([f"{word} ({weight})" for word, weight in word_weight_dict.items()])
        wc = WordCloud(width=wc_width, height=wc_height, font_path=font_path, background_color='white')
        wc.generate(words_str)
        plt.figure(figsize=(10, 10), facecolor=None)
        plt.imshow(wc)
        plt.axis("off")
        plt.tight_layout(pad=0)
        plt.savefig(f"主题为{num_topics}_{topic}.png", dpi=300, bbox_inches='tight', pad_inches=0)
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

def save_to_csv(start_value, interval, plexs, plexs_new, coherence):
    """
    生成DataFrame并保存到CSV文件中。
    """
    try:
        list_length = len(plexs)
        end_value = start_value + list_length * interval
        x = np.arange(start_value, end_value, interval)
        df = pd.DataFrame({"x": x, "plexs": plexs, "plexs_new": plexs_new, "conherence": coherence})
        df.to_csv('lda_perplexity.csv', index=False)
    except Exception as e:
        logger.error(f"保存CSV时发生错误: {e}")


# 主逻辑
def main():
    # 定义列表
    plexs = []
    plexs_new = []
    coherence = []
    home_path = os.getcwd()
    result_path = path.join(home_path, 'C01_LDA/result2/')
    os.chdir(result_path)
    save_path = path.join(home_path, 'C01_LDA/data2/')
    processed_content = load_processed_docs(PROCESSED_LANGUAGE, save_path)

    dictionary = gensim.corpora.Dictionary(processed_content)
    corpus = [dictionary.doc2bow(tmp) for tmp in processed_content]
    corpora.MmCorpus.serialize('corpus.mm', corpus)

    for num_topics in range(START_VALUE, N_MAX_TOPICS, LENGTH):
        lda_model, perplexity_value, coherence_value = get_lda_model(num_topics, corpus, dictionary, processed_content, PASSES)
        corpus = corpora.MmCorpus('corpus.mm')
        testset = [corpus[c*j] for c in range(int(corpus.num_docs/j))]
        prep = perplexity(lda_model, testset, dictionary, len(dictionary.keys()), num_topics)

        logger.info(f"主题数为{num_topics}模型建立完成, 困惑度为{perplexity_value}, 抽样为{j}时的困惑度为{prep}, 一致性为{coherence_value}, 最大迭代次数为{PASSES}")
        save_lda_visualization(lda_model, corpus, dictionary, num_topics)
        topic_word_dict = get_topic_word_distribution_gensim(lda_model, dictionary, N_TOPICS_WORD, num_topics)
        save_topic_word_dicts(topic_word_dict, num_topics)
        generate_and_save_wordclouds(topic_word_dict, num_topics)
        plexs.append(perplexity_value)
        plexs_new.append(prep)
        coherence.append(coherence_value)
        save_to_csv(start_value=START_VALUE, interval=len(plexs), plexs=plexs, plexs_new=plexs_new, coherence=coherence)
    graph_draw(list(range(START_VALUE, N_MAX_TOPICS, LENGTH)), plexs_new)


if __name__ == "__main__":
    main()
