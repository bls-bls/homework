import seaborn as sns
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
# 读取数据
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

# 找到与 Happiness_score 相关性最高的变量
happiness_idx = variables.index('Happiness_score')
correlation_with_happiness = correlation_matrix[happiness_idx]
max_corr_idx = np.argmax(np.abs(correlation_with_happiness[1:])) + 1
most_correlated_variable = variables[max_corr_idx]

# 绘制散点图
plt.figure()
plt.scatter(X[:, happiness_idx], X[:, max_corr_idx], c='blue')
plt.xlabel('Happiness Score')
plt.ylabel(most_correlated_variable)
plt.title(f'Scatter plot of Happiness Score vs {most_correlated_variable}')
plt.show()

print(f'The most correlated variable with Happiness_score is: {most_correlated_variable}')
print(f'The correlation value is: {correlation_with_happiness[max_corr_idx]}')