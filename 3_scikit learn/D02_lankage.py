import pandas as pd

# 读取表头数据
headers = ['word', 'index']
word_df = pd.read_csv('D:\\pythonProject5\\2_Word2vec\\02word2vec_index.csv', encoding='utf-8', names = headers)
varieties = word_df['word'].values

# 读取向量数据
seeds_df = pd.read_csv("D:\\pythonProject5\\2_Word2vec\\02word2vec.csv", encoding='utf-8', header=None)
Sample = seeds_df.values #将数据转换为数组

# 归一化(可选)
from sklearn.preprocessing import normalize
Sample = normalize(Sample)

from scipy.cluster.hierarchy import linkage, dendrogram
import matplotlib.pyplot as plt
plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号
# 层次聚类
mergings = linkage(Sample, method='complete')
# 树状图结果
plt.figure(figsize=(100, 100))
dendrogram(mergings, labels = varieties, leaf_rotation=3, leaf_font_size=8)
plt.show()

# 得到标签结果
from scipy.cluster.hierarchy import fcluster
labels = fcluster(mergings, t=0.75, criterion='distance')
df = pd.DataFrame({'labels':labels, 'varieties':varieties})
ct = pd.crosstab(df['labels'], df['varieties'])
print(df)
print(ct)

df.to_csv('D:/pythonProject5/3_scikit learn/03word_listage_index.csv', index=False, header=False, encoding='utf-8')
df.to_excel('D:/pythonProject5/3_scikit learn/03word_listage_index.xlsx', index=False, header=False)