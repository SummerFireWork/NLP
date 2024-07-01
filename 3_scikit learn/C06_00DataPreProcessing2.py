######################################################

"""
输入为txt
预处理模板，可以将结果分别处理为中英文结果
"""
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

# In[1]:
# prepare function
# 确保nltk资源已下载
# nltk.download('stopwords')
# nltk.download('wordnet')
# nltk.download('punkt')

output_path = '/home/summer/PythonProject/pythonProject5/3_scikit learn/C06_lda'
file_path = '/home/summer/PythonProject/pythonProject5/3_scikit learn/C06_lda'
os.chdir(output_path)
dic_file = "/home/summer/PythonProject/pythonProject5/3_scikit learn/C06_lda/stop_dic/dict.txt"
stop_file = "/home/summer/PythonProject/pythonProject5/3_scikit learn/C06_lda/stop_dic/stopwords.txt"
# 英文预处理函数
def preprocess_english(text):
    # 转换为小写，分词，去停用词，词形还原
    #小写化
    text = text.lower()
    #去除特殊标点
    for c in string.punctuation:
        text = text.replace(c, ' ')
    #分词
    wordLst = nltk.word_tokenize(text)
    #去除停用词
    filtered = [w for w in wordLst if w not in stopwords.words('english')]
    #仅保留名词或特定POS
    refiltered =nltk.pos_tag(filtered)
    filtered = [w for w, pos in refiltered if pos.startswith('NN')]
    #词干化
    ps = PorterStemmer()
    filtered = [ps.stem(w) for w in filtered]
    return filtered

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


def separate_cne(text):
    chinese_pattern = re.compile(r'[\u4e00-\u9fa5]+')
    english_pattern = re.compile(r'[a-zA-Z0-9]+')
    chinese_parts = chinese_pattern.findall(text)
    english_parts = english_pattern.findall(text)
    return ' '.join(chinese_parts), ' '.join(english_parts)


def load_and_process_file(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        chinese_docs = []
        english_docs = []
        for line in f:
            line = line.strip()
            chinese_text = separate_cne(line)[0]  # 保留中文部分
            english_text = separate_cne(line)[1]  # 保留中文部分
            # chinese_docs.append(' '.join(jieba.cut(chinese_text)))  # 使用jieba分词
            chinese_docs.append(' '.join(preprocess_chinese(chinese_text)))
            english_docs.append(' '.join(preprocess_english(english_text)))
        return chinese_docs, english_docs
# In[2]:
# 文件路径
file_path = '/home/summer/PythonProject/pythonProject5/3_scikit learn/C06_lda/data2/data.txt'
# 处理文件并分词
chinese_texts, english_texts= load_and_process_file(file_path)

# 文本预处理：去除空字符串，避免后续处理出错
chinese_texts = [text for text in chinese_texts if text] # 得到的中文文本，每行都是一个列表，每个列表元素是一个分词
english_texts = [text for text in english_texts if text] # 得到的英文文本，每行都是一个列表，每个列表元素是一个分词

# 合并处理结果（实际应用中可能需要更复杂的策略来整合中英文信息）
# 注意：此处简化处理，直接合并了中英文的分词结果，实际应用中可能需要根据模型支持情况调整
all_tokens = chinese_texts + english_texts


# In[3]:
#该区域仅首次运行，进行文本预处理，第二次运行起注释掉

save_path = '/home/summer/PythonProject/pythonProject5/3_scikit learn/C06_lda/data/'
with open(save_path + 'processed_english.txt', 'w') as f:
    for line in english_texts:
        f.write(line+'\n')

with open(save_path + 'processed_chinese.txt', 'w') as f:
    for line in chinese_texts:
        f.write(line+'\n')

with open(save_path + 'processed_all.txt', 'w') as f:
    for line in all_tokens:
        f.write(line+'\n')
# ==============================================================================
# 从第二次运行起，直接获取预处理过的docLst，前面load数据、预处理均注释掉
# processed_english = []
# with open(save_path + 'processed_english.txt', 'r') as f:
#     for line in f.readlines():
#         if line != '':
#             processed_english.append(line.strip())
#==============================================================================
