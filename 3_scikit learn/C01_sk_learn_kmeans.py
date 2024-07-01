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
data = pd.read_csv('D:\\pythonProject5\\3_scikit learn\\iris.csv')
# 定义 iris 的三个品种作为标签
iris_types = ['virginica', 'setosa', 'versicolor']

# 定义 x 轴和 y 轴的特征
x_axis = 'Petal.Length'
y_axis = 'Petal.Width'

# 创建一个大的图像窗口，并划分两个子图
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)  # 第一个子图，已知标签的散点图
# 根据标签颜色区分不同品种的 iris
for iris_type in iris_types:
    plt.scatter(data[x_axis][data['Species'] == iris_type], data[y_axis][data['Species'] == iris_type], marker='o', label=iris_type)
plt.title("label know")  # 图表标题
plt.legend()  # 显示图例

plt.subplot(1, 2, 2)  # 第二个子图，未知标签的散点图
plt.scatter(data[x_axis][:], data[y_axis][:], marker='o')
plt.title("label unknow")  # 图表标题
plt.show()  # 显示图像

### 聚类分析
# 创建 KMeans 模型，并指定聚类的类别数量为 3
model = KMeans(n_clusters=3)
# 使用模型对数据集进行聚类分析
model.fit(data[[x_axis, y_axis]])
# 获取聚类结果标签
labels = model.labels_
print(labels)
# 将列表labels导出为 csv 文件
pd.DataFrame(labels).to_csv('D:\\pythonProject5\\3_scikit learn\\labels.csv', index=False, header=False)
# 获得聚类中心
centers = model.cluster_centers_
print(centers)