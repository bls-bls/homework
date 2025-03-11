import seaborn as sns
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
# 读取数据
# 读取数据
from mpl_toolkits.mplot3d import Axes3D
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

# 读取数据
data = pd.read_csv(r'2024_happy.csv')

# 选择变量
variables = data.columns[5:]
X = data.iloc[:, 5:].values

# 删除含有缺失值的行
X_clean = X[~np.isnan(X).any(axis=1)]

# 数据标准化
scaler = StandardScaler()
X_standardized = scaler.fit_transform(X_clean)

# PCA
pca = PCA(n_components=3)
score = pca.fit_transform(X_standardized)

# K-means 聚类
kmeans = KMeans(n_clusters=3)
idx = kmeans.fit_predict(score)
centers = kmeans.cluster_centers_

# 绘制 3D 图
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(score[idx == 0, 0], score[idx == 0, 1], score[idx == 0, 2], c='r', marker='o', s=50)
ax.scatter(score[idx == 1, 0], score[idx == 1, 1], score[idx == 1, 2], c='g', marker='o', s=50)
ax.scatter(score[idx == 2, 0], score[idx == 2, 1], score[idx == 2, 2], c='b', marker='o', s=50)
ax.scatter(centers[:, 0], centers[:, 1], centers[:, 2], c='k', marker='x', s=200, linewidths=3)

# 添加国家名称
country_names = data['Country_name'][~np.isnan(X).any(axis=1)].values
for i, txt in enumerate(country_names):
    ax.text(score[i, 0], score[i, 1], score[i, 2], txt, fontsize=8, ha='right')

# 设置标题和标签
ax.set_title('K-means Clustering')
ax.set_xlabel(f'Principal Component 1 ({variables[0]})')
ax.set_ylabel(f'Principal Component 2 ({variables[1]})')
ax.set_zlabel(f'Principal Component 3 ({variables[2]})')

# 添加图例
ax.legend(['Cluster 1', 'Cluster 2', 'Cluster 3', 'Cluster Centers'])
plt.show()