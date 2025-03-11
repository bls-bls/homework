import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

# 读取数据
data = pd.read_csv(r'2024_happy.csv')

# 获取变量名称
variables = data.columns[5:]

# 提取数据
X = data.iloc[:, 5:].values

# 检查缺失值
print(np.isnan(X).any(axis=0))  # 每列是否有缺失值
print(np.isnan(X).any(axis=1))  # 每行是否有缺失值

# 删除含有缺失值的行
X_clean = X[~np.isnan(X).any(axis=1)]

# 数据标准化
scaler = StandardScaler()
X_standardized = scaler.fit_transform(X_clean)

# PCA
pca = PCA()
score = pca.fit_transform(X_standardized)
explained = pca.explained_variance_ratio_
cumulative_explained = np.cumsum(explained)

# 绘制柱状图和折线图
plt.figure()
plt.bar(range(1, len(explained) + 1), explained, color='blue', edgecolor='black', linewidth=1)
plt.plot(range(1, len(cumulative_explained) + 1), cumulative_explained, 'r-o', color='red', linewidth=1.5)
plt.title('PCA of Happiness Score: Explained Variance and Cumulative Contribution')
plt.xlabel('Principal Component')
plt.ylabel('Variance Explained (%)')
plt.legend(['Explained Variance', 'Cumulative Contribution'])
plt.xticks(range(1, len(explained) + 1), variables[:len(explained)])
plt.show()