"""
该脚本用于可视化 iris 数据集，并对其进行 KMeans 聚类分析。
数据集来源于 scikit-learn 库中的 iris 数据。
可视化分为两部分：已知标签的散点图和未知标签的散点图。
"""

# 导入必要的库
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans


### 可视化
# 从指定路径读取 iris 数据集
data = pd.read_csv('D:\\pythonProject5\\2_Word2vec\\word2vec.csv', encoding='utf-8', header=None)
# 选择两个特征作为 x 和 y 轴
data1 = np.array(data[0])
data2 = np.array(data[1])
X = np.vstack((data1, data2)).T
print(X)
### 聚类分析
# 创建 KMeans 模型，并指定聚类的类别数量为 3
model = KMeans(n_clusters=3)
# 使用模型对数据集进行聚类分析
model.fit(X)
# 获取聚类结果标签
labels = model.labels_
print(labels)
# 将列表labels导出为 csv 文件
pd.DataFrame(labels).to_csv('D:\\pythonProject5\\3_scikit learn\\labels.csv', index=False, header=False)
# 获得聚类中心
centers = model.cluster_centers_
print(centers)
