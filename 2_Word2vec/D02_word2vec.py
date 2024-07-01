"""
注意，停用词在D:\pythonProject5\stopwords当中，可自行进行修改。
维度修改在第119行，如果需要则进行修改

"""
import matplotlib.pyplot as plt

"""
第一步：本程序目的是对中文文件进行分词，仅需要修改输入、输出文件即可。
"""
# 导入jieba模块及其扩展
import jieba
import jieba.analyse
import jieba.posseg as pseg
import codecs, sys
import re
import os


def get_stop_dict(file):
    content = open(file,encoding="utf-8")
    word_list = []
    for c in content:
        c = re.sub('\n|\r','',c)
        word_list.append(c)
    return word_list


# 定义工作目录路径
os.chdir("\\\\wsl$\\Ubuntu-24.04\\home\\summer\\PythonProject\\pythonProject5")
# 定义停用词文件路径
stop_file1 = 'stopwords/baidu_stopwords.txt'
stop_file2 = 'stopwords/cn_stopwords.txt'
stop_file3 = 'stopwords/hit_stopwords.txt'
stop_file4 = 'stopwords/scu_stopwords.txt'
stop_file5 = 'stopwords/own_stopwords.txt'
# 定义用户自定义词典文件路径
user_file = 'stopwords/add_word_list.txt'
# 定义待处理文本文件路径
input_file = "2_Word2vec/01theFirstOne.txt"
output_file = "2_Word2vec/01theFirstOne.target.txt"
# 词向量维度
dimension = 100
# 输出维度
output_dimension = 2
# 输出词向量
output_xlsx_vector = "2_Word2vec/02word2vec.csv"
# 输出词排序的列表
output_xlsx = "2_Word2vec/02word2vec_index.csv"
# 定义需要忽略的词性标记列表
flag_list = ['n','nz','vn']
whether_chinese = True


# 打开输入与输出文件
f = codecs.open(input_file, "r", "utf-8")
target = codecs.open(output_file, "w", "utf-8")

# 获取停用词列表
stop_words = get_stop_dict(stop_file1)+get_stop_dict(stop_file2)+get_stop_dict(stop_file3)+get_stop_dict(stop_file4)+get_stop_dict(stop_file5)
# 加载用户自定义词典
jieba.load_userdict(user_file)

# 对文件中的每一行进行分词，并将结果写入输出文件
line_num = 1  # 行号初始化
line = f.readline()  # 读取文件第一行
while line:
    print("line:", line_num)  # 打印当前处理的行号
    # line_seg = " ".join(jieba.cut(line))  # 对当前行进行分词，旧版本1，未增加停用词
    # line_seg = " ".join(word for word in jieba.cut(line) if word not in stop_words and len(word)>1) # 旧版本2，未处理4.4这样的数字字符
    words1 = [word for word in jieba.cut(line) if word not in stop_words and len(word)>1]
    line_seg = " ".join(word for word in words1 if not word.isdigit() and not any(char.isdigit() for char in word))
    if whether_chinese:
        for word in line_seg: # 去除英文词语（可选）
            if not re.match(r'[\u4e00-\u9fa5]+', word):
                line_seg = line_seg.replace(word, ' ')
    print(line_seg)
    target.writelines(line_seg)  # 将分词结果写入输出文件
    line_num = line_num + 1  # 行号自增
    line = f.readline()  # 读取下一行
f.close()  # 关闭输入文件
target.close()  # 关闭输出文件

#####################################################################################################################################

"""
第二步，训练模型
"""
import logging
import os.path
import sys
import multiprocessing
from gensim.models import Word2Vec
from gensim.models.word2vec import LineSentence


# 使用WikiCorpus和Word2Vec训练模型
# 参数包括：输入文件路径，词向量大小，窗口大小，出现次数最少的词频，使用的worker线程数
model = Word2Vec(LineSentence(output_file), vector_size=dimension, window=5, min_count=5, workers=multiprocessing.cpu_count())

# 保存Word2Vec模型
model.save('2_Word2vec/01theFirstOne.model')

# 以非二进制格式保存词向量
model.wv.save_word2vec_format('2_Word2vec/01theFirstOne.vector', binary=False)


#############################################################################################################################################
"""
第三步，使用模型
"""
from gensim.models import Word2Vec

bridge_model = Word2Vec.load('2_Word2vec/01theFirstOne.model')

testword = ['中国', '美国', '日本']
for i in range(len(testword)):
    if testword[i] not in bridge_model.wv.index_to_key:
        print('该词不在词表中')
        continue
    res = bridge_model.wv.most_similar(testword[i])
    print(testword[i])
    print(res)

"""
2. 模型可视化-可选,将结果输出到origin中绘制
rawWordVec 向量列表
word2ind[w] 词表
X_reduced 降维后的词向量
"""
import numpy as np
from sklearn.decomposition import PCA

# 将词向量投影到二维空间
rawWordVec = [] # 词向量
word2ind = {} # 词表
for j, w in enumerate(bridge_model.wv.index_to_key):
    rawWordVec.append(bridge_model.wv[w])  # 词向量
    word2ind[w] = j  # {词语:序号}
# print("词向量维度：", bridge_model.wv.vectors.shape)
# print("词表：", word2ind.keys())
# print("词向量：", rawWordVec)
rawWordVec = np.array(rawWordVec)
X_reduced = PCA(n_components=output_dimension).fit_transform(rawWordVec)  # PCA降2维

plt.figure(figsize=(10, 10))
plt.scatter(X_reduced[:, 0], X_reduced[:, 1])
plt.show()
# /**
# * 将X_reduced(向量)输出到csv文件中
# */

import pandas as pd
df = pd.DataFrame(X_reduced)
df.to_csv(output_xlsx_vector, index=False, header=False)

# /**
# * 将word2ind的keys 和value 输出到csv文件中
# */
df2 = pd.DataFrame(list(word2ind.items()), columns=['word', 'index'])
df2.to_csv(output_xlsx, index=False, header=False)

"""
进行层次聚类
"""
with open("D:\\pythonProject5\\3_scikit learn\\D02_lankage.py", encoding="utf-8") as file:
    exec(file.read())