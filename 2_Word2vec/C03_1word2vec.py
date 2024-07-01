"""
该程序用于对指定文本文件进行分词，使用结巴中文分词库进行词语切分，并通过正则表达式去除标点符号。
运行此程序需要预先安装 jieba 和 re 模块。
是在pycharm中才可以运行

不接受任何参数。
返回值: 无。程序直接将处理结果打印输出。
"""

import jieba
import re
import warnings

# 忽略所有警告
warnings.filterwarnings('ignore')

# 1.文本处理——读入文本文件
with open("D:\\pythonProject5\\2_Word2vec\\The praise of folly_text.txt", 'r', encoding='utf-8') as f:
    lines = []
    for line in f:  # 对文件的每一行进行处理
        temp = jieba.lcut(line)  # 使用结巴分词进行词语切分
        words = []
        for i in temp:
            # 使用正则表达式去除每个词中的所有标点符号
            i = re.sub("[\s+\.\!\/_,$%^*(+\"\'””《》]+|[+——！，。？、~@#￥%……&*（）：；‘]+", "", i)
            if len(i) > 0:  # 如果处理后的词非空，则加入到结果列表中
                words.append(i)
        if len(words) > 0:  # 如果当前行存在有效词，则加入到行列表中
            lines.append(words)
    # 打印处理后的文本的前5行
print(lines[0:5])

# 2.模型建立
from gensim.models import Word2Vec
# 调用Word2Vec训练 参数：size: 词向量维度；window: 上下文的宽度，min_count为考虑计算的单词的最低词频阈值
model = Word2Vec(lines,vector_size = 20, window = 2 , min_count = 3, epochs=7, negative=10,sg=1)
print("我 的词向量：\n",model.wv.get_vector('我'))
print("\n和我相关性最高的前20个词语：")
print(model.wv.most_similar('我', topn = 20))# 与孔明最相关的前20个词语


# 3.模型可视化
import numpy as np
from sklearn.decomposition import PCA

# 将词向量投影到二维空间
rawWordVec = [] # 词向量
word2ind = {} # 词表
for i, w in enumerate(model.wv.index_to_key):
    rawWordVec.append(model.wv[w])  # 词向量
    word2ind[w] = i  # {词语:序号}
print("词向量维度：", rawWordVec.shape)
print("词表：", word2ind.keys())
print("词向量：", rawWordVec)
rawWordVec = np.array(rawWordVec)
X_reduced = PCA(n_components=2).fit_transform(rawWordVec)  # PCA降2维

import matplotlib.pyplot as plt

plt.rcParams['font.sans-serif'] = ['SimHei']  # 解决中文显示
plt.rcParams['axes.unicode_minus'] = False  # 解决符号无法显示
# 绘制星空图
# 绘制所有单词向量的二维空间投影
fig = plt.figure(figsize=(15, 10))
ax = fig.gca()
ax.set_facecolor('white')
ax.plot(X_reduced[:, 0], X_reduced[:, 1], '.', markersize=1, alpha=0.3, color='black')

# 绘制几个特殊单词的向量
words = ['孙权', '刘备', '曹操', '周瑜', '诸葛亮', '司马懿', '汉献帝']

for w in words:
    if w in word2ind:
        ind = word2ind[w]
        xy = X_reduced[ind]
        plt.plot(xy[0], xy[1], '.', alpha=1, color='orange', markersize=10)
        plt.text(xy[0], xy[1], w, alpha=1, color='red')

plt.show()

# 4.进行好玩的测试
# 玄德－孔明＝？－曹操
# words = model.wv.most_similar(positive=['玄德', '曹操'], negative=['孔明'])
# 我-人们 =？ - 别人
print("\n进行测试：")
words = model.wv.most_similar(positive=['我', '别人'], negative=['人们'])
print(words)