import pandas as pd

seeds_df = pd.read_csv("D:\\pythonProject5\\3_scikit learn\\iris.csv", encoding='utf-8')

varieties = seeds_df["Species"].values
seeds_df.pop("Species") #删除列最后一个
Sample = seeds_df.values #将数据转换为数组

from sklearn.preprocessing import normalize
Sample = normalize(Sample)
from scipy.cluster.hierarchy import linkage, dendrogram
import matplotlib.pyplot as plt

# 层次聚类
mergings = linkage(Sample, method='complete')
# 树状图结果
plt.figure(figsize=(10, 10))
dendrogram(mergings, labels=varieties, leaf_rotation=90, leaf_font_size=8)
plt.show()

# 得到标签结果
from scipy.cluster.hierarchy import fcluster
labels = fcluster(mergings, t=65, criterion='distance')
df = pd.DataFrame({'labels':labels, 'varieties':varieties})
ct = pd.crosstab(df['labels'], df['varieties'])
print(df)
print(ct)
