# Import necessary libraries
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs

"""
This script demonstrates the K-means clustering algorithm using scikit-learn.
"""
# Generate artificial blob data for clustering visualization
blob_centers = np.array([[0.2, 2.4], [-1, 1.3], [-2.8, 0], [1.1, 2.3], [-2.9, 2.1]])  # Centers of the blobs
blob_std = np.array([0.4, 0.3, 0.1, 0.1, 0.1])  # Standard deviations of the blobs
X, y = make_blobs(n_samples=2000, centers=blob_centers, cluster_std=blob_std, random_state=7)  # Generate the blob dataset

def plot_clusters(X, y=None):
    """
    Plot the given data points as clusters.

    Parameters:
    X (numpy.ndarray): The data points to be plotted, where each row represents a point.
    y (numpy.ndarray, optional): The labels for the data points. If not provided, points are plotted without labels.

    Returns:
    None: This function does not return any value. It displays a scatter plot.
    """
    # Plot the data points with colors based on their labels
    plt.scatter(X[:, 0], X[:, 1], c=y, s=1)
    plt.xlabel("$x_1$", fontsize=14)
    plt.ylabel("$x_2$", fontsize=14, rotation=0)

# Create a new figure and plot the clusters
plt.figure(figsize=(8, 4))
plot_clusters(X)
# plt.show()

"""
Apply K-means clustering to the data
"""

from sklearn.cluster import KMeans
# 初始化KMeans聚类模型，设定聚类中心个数为k，随机状态为42以确保结果可复现
k = 5
kmeans = KMeans(n_clusters=k, random_state=42)
# 对数据X进行聚类，并返回聚类结果
y_pred = kmeans.fit_predict(X)
# 打印预测的聚类标签
print(y_pred)
# 打印模型分配的聚类标签
print(kmeans.labels_)
# 打印每个聚类中心的坐标
print(kmeans.cluster_centers_)

"""
# 这段代码用于对给定的新数据点进行KMeans聚类预测
# X_new: 新数据点的集合，每个数据点包含两个特征
# 返回值: 预测的类别标签列表
"""
X_new = np.array([[0, 2], [3, 2], [-3, 3], [-3, 2.5]])  # 定义新数据点集合

# 对新数据点进行聚类预测
kmeans.predict(X_new)
print(kmeans.predict(X_new))  # 打印预测结果
print(kmeans.transform(X_new)) # 计算每个新点与中心坐标的距离
print(kmeans.inertia_)

"""
可视化函数
"""
def plot_data(X):
    """
    Plot the given data points.

    Parameters:
    X (numpy.ndarray): The data points to be plotted, where each row represents a point.

    Returns:
    None: This function does not return any value. It displays a scatter plot.
    """
    # Plot the data points
    plt.scatter(X[:, 0], X[:, 1], s=1)
    plt.xlabel("$x_1$", fontsize=14)
    plt.ylabel("$x_2$", fontsize=14, rotation=0)

# 创建一个新图，并绘制数据点
plt.figure(figsize=(8, 4))
plot_data(X)
plt.show()

