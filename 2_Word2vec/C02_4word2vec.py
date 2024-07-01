# -*- coding: utf-8 -*-

"""
第三步，使用模型
"""
from gensim.models import Word2Vec

bridge_model = Word2Vec.load('01theFirstOne.model')

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
X_reduced = PCA(n_components=2).fit_transform(rawWordVec)  # PCA降2维

# /**
# * 将X_reduced(向量)输出到csv文件中
# */

import pandas as pd
df = pd.DataFrame(X_reduced)
df.to_csv('02word2vec.csv', index=False, header=False)

# /**
# * 将word2ind的keys 和value 输出到csv文件中
# */
df2 = pd.DataFrame(list(word2ind.items()), columns=['word', 'index'])
df2.to_csv('02word2vec_index.csv', index=False, header=False)