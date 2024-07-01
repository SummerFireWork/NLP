import pandas as pd
from sklearn.cluster import KMeans
import matplotlib as plt

data = pd.read_excel("D:\\知识文件\\数据分析\\B11SPSS\\大学课程《统计分析与SPSS的应用》\\大学生职业生涯规划.xls")
# print(data.to_string)
print(data.columns)

# 选择聚类的列
col = ['Q1','Q2','Q3','Q4']
print(data[col])
data[col].fillna(value=1, inplace=True)

SSE = []
min_n = 2
max_n = 10
for k in range(min_n, max_n):
    km = KMeans(n_clusters=k)
    km.fit(data[col])
    SSE.append(km.inertia_)

X = range(min_n, max_n)
plt.xlabel('k')
plt.ylabel('SSE')
plt.plot(X, SSE, '0-')
plt.show()