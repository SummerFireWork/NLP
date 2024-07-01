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

# 提前编译正则表达式，避免在循环中重复编译
chinese_pattern = re.compile(r'[\u4e00-\u9fa5]+')
english_pattern = re.compile(r'[a-zA-Z]+')

# In[1]:
# prepare function
# 确保nltk资源已下载
# try:
#     nltk.download('stopwords')
#     nltk.download('wordnet')
#     nltk.download('punkt')
# except:
#     pass

# get data directory (using getcwd() is needed to support running example in generated IPython notebook)
home_path = os.getcwd()
# Read the whole text.
output_path = path.join(home_path, 'C01_LDA')
dic_file = path.join(home_path, 'C01_LDA/stop_dic/dict.txt')
stop_file = path.join(home_path, 'C01_LDA/stop_dic/stopwords.txt')
file_path = path.join(home_path, 'C01_LDA/data/data.txt')
save_path = path.join(home_path, 'C01_LDA/data/')


# In[2]:
# 共享的预处理步骤
def shared_preprocess(text):
    # 转换为小写，去除特殊标点
    text = text.lower()
    for c in string.punctuation:
        text = text.replace(c, ' ')
    return text


# 英文预处理函数
def preprocess_english(text):
    # 获取停用词
    stop_words = load_stop_words(stop_file)
    # 使用共享的预处理
    text = shared_preprocess(text)
    # 分词
    wordLst = nltk.word_tokenize(text)
    # 去除停用词
    filtered = [w for w in wordLst if w not in stop_words]
    # 仅保留名词或特定POS
    refiltered = nltk.pos_tag(filtered)
    filtered = [w for w, pos in refiltered if pos.startswith('NN')]
    # 词干化
    ps = PorterStemmer()
    filtered = [ps.stem(w) for w in filtered]
    return filtered


# 中文预处理函数，使用jieba分词
def preprocess_chinese(text):
    jieba.load_userdict(dic_file)
    jieba.initialize()
    # 获取停用词
    stop_words = load_stop_words(stop_file)

    flag_list = ['n', 'nz', 'vn']
    word_list = []
    seg_list = psg.cut(text)
    for seg_word in seg_list:
        word = seg_word.word
        if word not in stop_words and len(word) >= 2 and seg_word.flag in flag_list:
            word_list.append(word)
    return word_list


def separate_cne(text):
    chinese_parts = chinese_pattern.findall(text)
    english_parts = english_pattern.findall(text)
    return ' '.join(chinese_parts), ' '.join(english_parts)


def load_and_process_file(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        chinese_docs = []
        english_docs = []
        for line in f:
            line = line.strip()
            chinese_text, english_text = separate_cne(line)
            chinese_docs.append(' '.join(preprocess_chinese(chinese_text)))
            english_docs.append(' '.join(preprocess_english(english_text)))
        return chinese_docs, english_docs


def load_stop_words(stop_file):
    stop_list = set()
    stop_list_final = []
    try:
        if not os.path.exists(stop_file):
            raise FileNotFoundError(f"The stop words file {stop_file} does not exist.")

        with open(stop_file, encoding='utf-8') as stopword_list:
            for line in stopword_list:
                line = line.strip()
                stop_list.add(line)

            # 将标准库中的英语停用词也转换为set，以提高合并效率
            english_stop_words = set(stopwords.words('english'))
            stop_list_final = list(stop_list.union(english_stop_words))

    except FileNotFoundError as fnf_error:
        print(f"Error loading stop words: {fnf_error}")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")

    return stop_list_final


def main():

    # 处理文件并分词
    chinese_texts, english_texts = load_and_process_file(file_path)
    # 文本预处理：去除空字符串，避免后续处理出错
    chinese_texts = [text for text in chinese_texts if text]  # 得到的中文文本，每行都是一个列表，每个列表元素是一个分词
    english_texts = [text for text in english_texts if text]  # 得到的英文文本，每行都是一个列表，每个列表元素是一个分词

    # 合并处理结果（实际应用中可能需要更复杂的策略来整合中英文信息）
    # 注意：此处简化处理，直接合并了中英文的分词结果，实际应用中可能需要根据模型支持情况调整
    all_tokens = chinese_texts + english_texts

    # 该区域仅首次运行，进行文本预处理，第二次运行起注释掉

    with open(save_path + 'processed_english.txt', 'w') as f:
        for line in english_texts:
            f.write(line + '\n')

    with open(save_path + 'processed_chinese.txt', 'w') as f:
        for line in chinese_texts:
            f.write(line + '\n')

    with open(save_path + 'processed_all.txt', 'w') as f:
        for line in all_tokens:
            f.write(line + '\n')


if __name__ == "__main__":
    main()
