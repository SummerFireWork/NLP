"""
分词程序，中文分词，会将英文删去，如果想要修改，建议找英文教程，或修改第62行代码
本分词程序基于jieba库
"""

import os
import jieba
import jieba.posseg as psg
import re
import pandas as pd

# 定义工作目录路径
file_path = 'D:\\pythonProject5\\1_Word_Segmentation\\D01_wordfreq_zh'
# 定义停用词文件路径
stop_file = 'stopwordlist.txt'
# 定义用户自定义词典文件路径
user_file = 'add_word_list.txt'
# 定义待处理文本文件路径
file_name = 'test.txt'
# 定义需要忽略的词性标记列表
flag_list = ['n','nz','vn']

def get_stop_dict(file):
    """
    读取停用词文件并返回停用词列表。

    :param file: 停用词文件路径
    :return: 停用词列表
    """
    content = open(file, encoding="utf-8")
    word_list = []
    for c in content:
        c = re.sub('\n|\r', '', c)
        word_list.append(c)
    return word_list


# 改变当前工作目录到指定路径
os.chdir(file_path)
# 从停用词文件中获取停用词字典
stop_words = get_stop_dict(stop_file)
# 打开文本文件并以UTF-8编码读取内容
text = open(file_name,encoding="utf-8").read()
# 加载用户自定义词典
jieba.load_userdict(user_file)
# 将文本内容按行分割成列表
text_lines  = text.split('\n')


# 初始化词频统计字典
counts={}




# 对文本行进行分词和筛选，统计高频词
for line in text_lines:
    # 使用分词工具对当前行进行分词
    line_seg = psg.cut(line)
    for word_flag in line_seg:
        # 移除非中文字符，确保词是纯中文
        word = re.sub("[^\u4e00-\u9fa5]","",word_flag.word)
        # 筛选符合要求的词：在标志列表中、长度大于1、不在停用词列表中
        if word_flag.flag in flag_list and len(word)>1 and word not in stop_words:
            # 对符合条件的词进行计数
            counts[word]=counts.get(word,0)+1

# 将词频统计结果转换为DataFrame格式，准备进行排序和输出
word_freq = pd.DataFrame({'word':list(counts.keys()),'freq':list(counts.values())})
# 按词频降序排列
word_freq = word_freq.sort_values(by='freq',ascending=False)
# 将排序后的词频结果导出到Excel文件
word_freq.to_excel("01word_freq.xlsx",index=False)

"""
合并同义词
"""

df = pd.read_excel('01word_freq.xlsx')
syn_name = 'synonym.txt'
#每行为互为同义词的几个词语，空格隔开(教师 老师 教授)，行首的词语为最终替换词语(最终全部合并为“教师”)
txt = open(syn_name,encoding="utf-8").read()
txts = txt.split("\n")

for line in txts:
    words = line.split(" ")
    dic = {}
    for word in words:
        dic[word]=words[0]
    df['word']=df['word'].replace(dic)

df['new_freq']=df.groupby(['word'], as_index=False).cumsum()
df = df.drop_duplicates(subset=['word'], keep='last')
df=df[['word','new_freq']]
df.to_excel("02word_freq_combine_synonym.xlsx",index=False)#保存新的词频文件
