"""
输入为txt
预处理模板，可以将结果分别处理为中英文结果
"""

import re
import jieba
import jieba.analyse
import jieba.posseg as psg
from gensim import corpora, models
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import nltk

# 确保nltk资源已下载
nltk.download('stopwords')
nltk.download('wordnet')

dic_file = "/8_genmis/C01_LDA/stop_dic/dict.txt"
stop_file = "/8_genmis/C01_LDA/stop_dic/stopwords.txt"
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
def preprocess_chinese2(text):
    # 使用jieba分词，默认模式
    tokens = jieba.lcut(text)
    # 可以选择性地去除中文停用词，这里省略以保持示例简单
    print(tokens)
    print(type(tokens))
    return tokens

# 中文预处理
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

# 示例文档
doc = "这是一个中英文混杂的例子。This is an example of mixed Chinese and English content."
with open('/8_genmis/C01_LDA/data/data.txt', 'r', encoding='utf-8') as f:
    doc = f.read()
    f.close()

# 分离中英文
chinese_text, english_text = separate_cne(doc)

# 预处理
processed_chinese = preprocess_chinese(chinese_text)
processed_english = preprocess_english(english_text)

# 合并处理结果（实际应用中可能需要更复杂的策略来整合中英文信息）
# 注意：此处简化处理，直接合并了中英文的分词结果，实际应用中可能需要根据模型支持情况调整
all_tokens = processed_chinese + processed_english

# 以下部分与之前英文LDA处理流程相同，但需注意，混合语言的LDA可能需要更专业的模型或库支持
# （以下代码未完成，仅示意）
# dictionary = corpora.Dictionary([all_tokens])
# corpus = [dictionary.doc2bow(all_tokens)]
# lda_model = models.LdaModel(corpus, num_topics=num_topics, id2word=dictionary, passes=15)

print("Processed Chinese Tokens:", processed_chinese)
print("Processed English Tokens:", processed_english)