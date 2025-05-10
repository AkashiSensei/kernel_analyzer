import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from matplotlib.colors import Normalize, FuncNorm

# conv
data = {'规则库数据来源计算图': ['V8n', 'V8s', 'V8m', 'V8l', 'V8x'], 'V8n': ['64/64', '49/64', '41/64', '47/64', '42/64'], 'V8s': ['30/64', '64/64', '52/64', '46/64', '45/64'], 'V8m': ['29/84', '67/84', '84/84', '69/84', '62/84'], 'V8l': ['39/104', '60/104', '90/104', '104/104', '93/104'], 'V8x': ['40/104', '57/104', '75/104', '92/104', '104/104']}

# 创建 DataFrame
df = pd.DataFrame(data)
df.set_index('规则库数据来源计算图', inplace=True)

# 将分数转换为百分比
def convert_to_percentage(value):
    numerator, denominator = map(int, value.split('/'))
    return numerator / denominator * 100

for col in df.columns:
    df[col] = df[col].apply(convert_to_percentage)

# 设置字体
plt.rcParams['font.family'] = 'SimSun'
# 增大字体大小
plt.rcParams['font.size'] = 28

# 定义 100% 时的颜色
max_color = '#0d2d2a'

# 创建颜色映射
cmap = sns.light_palette(max_color, as_cmap=True)

# 指定最浅色对应的值
min_value = 30
max_value = 100

# 创建自定义的归一化对象
norm = Normalize(vmin=min_value, vmax=max_value)

# 调整图形尺寸，可根据实际情况调整
plt.figure(figsize=(10, 8))

# 绘制热图，同时调整注释字体大小和加粗，精细调整块之间的间距
# 调整颜色条与图的间距
sns.heatmap(df, annot=True, fmt=".1f", cmap=cmap, linewidths=5, annot_kws={"size": 28, "weight": "bold"},
            cbar_kws={"pad": 0.03}, norm=norm)

# 添加横轴标题，并设置加粗，调整横轴标题和标签的距离
plt.xlabel('被预测目标计算图', fontweight='bold', labelpad=15)
# 设置纵轴标签并加粗，调整纵轴标题和标签的距离
plt.ylabel('规则库数据来源计算图', fontweight='bold', labelpad=15)

# 调整横轴刻度标签与图的间距
plt.tick_params(axis='x', pad=10)
# 调整纵轴刻度标签与图的间距
plt.tick_params(axis='y', pad=10)

# 调整子图布局，为标题和坐标轴留出更多空间
plt.subplots_adjust(left=0.13, bottom=0.16, right=0.96, top=0.95)

# 移除大标题
# 保存图片，设置 dpi 提高清晰度
plt.savefig('./plotter/output/yolov8n_to_x_conv_kernels_accuracy.png', dpi=100)
plt.close()
    