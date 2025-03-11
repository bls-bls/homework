import seaborn as sns
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
# 读取数据
data = pd.read_csv(r'2024_happy.csv')

# 选择变量
variables = ['Happiness_score', 'Economy_GDP_per_Capita', 'Social_support',
             'Healthy_life_expectancy', 'Freedom_to_make_life_choices',
             'Generosity', 'Perceptions_of_corruption']
X = data[variables].values

# 删除含有缺失值的行
X = X[~np.isnan(X).any(axis=1)]

# 计算相关性矩阵
correlation_matrix = np.corrcoef(X, rowvar=False)

# 绘制热力图
plt.figure()
sns.heatmap(correlation_matrix, annot=True, xticklabels=variables, yticklabels=variables, cmap='jet')
plt.title('Correlation Heatmap of the Variables')
plt.show()