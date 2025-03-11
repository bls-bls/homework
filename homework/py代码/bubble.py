import plotly.express as px
import pandas as pd

# 读取 CSV 文件
df = pd.read_csv("2024_happy.csv")

# 绘制热力图
fig = px.choropleth(
    df,
    locations="Country_name",  # 国家名称
    locationmode="country names",  # 使用国家名称
    color="Happiness_score",  # 颜色映射的数值（幸福指数）
    hover_name="Country_name",  # 悬停显示的国家名称
    hover_data={"Happiness_score": True},  # 悬停时显示幸福指数
    color_continuous_scale=px.colors.sequential.Plasma,  # 颜色方案
    title="World Happiness Index 2024"  # 图表标题
)

# 显示图表
fig.show()