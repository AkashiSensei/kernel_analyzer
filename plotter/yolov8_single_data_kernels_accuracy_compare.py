import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from matplotlib.colors import Normalize, FuncNorm

# 精确数据
data = {
    '规则库数据来源模型': ['V8n', 'V8s', 'V8m', 'V8l', 'V8x'],
    'V8n': ['239/239', '146/239', '94/239', '119/239', '54/239'],
    'V8s': ['115/239', '239/239', '69/239', '146/239', '37/239'],
    'V8m': ['50/305', '51/305', '305/305', '51/305', '37/305'],
    'V8l': ['81/371', '246/371', '51/371', '371/371', '37/371'],
    'V8x': ['104/371', '37/371', '37/371', '37/371', '371/371']
}

# 序列数据
data = {
    '规则库数据来源模型': ['V8n', 'V8s', 'V8m', 'V8l', 'V8x'],
    'V8n': ['239/239', '224/239', '210/239', '216/239', '211/239'],
    'V8s': ['205/239', '239/239', '221/239', '215/239', '214/239'],
    'V8m': ['248/305', '286/305', '305/305', '288/305', '281/305'],
    'V8l': ['298/371', '319/371', '349/371', '371/371', '360/371'],
    'V8x': ['299/371', '316/371', '334/371', '359/371', '371/371']
}

# 重做后的数据
#   精确
data = {'规则库数据来源模型': ['V8n', 'V8s', 'V8m', 'V8l', 'V8x'], 'V8n': ['239/239', '146/239', '94/239', '119/239', '54/239'], 'V8s': ['115/239', '239/239', '69/239', '146/239', '37/239'], 'V8m': ['50/305', '51/305', '305/305', '51/305', '37/305'], 'V8l': ['81/371', '247/371', '51/371', '371/371', '37/371'], 'V8x': ['104/371', '37/371', '37/371', '37/371', '371/371']}
#   序列
# data = {'规则库数据来源模型': ['V8n', 'V8s', 'V8m', 'V8l', 'V8x'], 'V8n': ['239/239', '224/239', '216/239', '222/239', '217/239'], 'V8s': ['205/239', '239/239', '227/239', '221/239', '220/239'], 'V8m': ['250/305', '288/305', '305/305', '290/305', '283/305'], 'V8l': ['306/371', '327/371', '357/371', '371/371', '360/371'], 'V8x': ['307/371', '324/371', '342/371', '359/371', '371/371']}

# 创建 DataFrame
df = pd.DataFrame(data)
df.set_index('规则库数据来源模型', inplace=True)

# 将分数转换为百分比
def convert_to_percentage(value):
    numerator, denominator = map(int, value.split('/'))
    return numerator / denominator * 100

for col in df.columns:
    df[col] = df[col].apply(convert_to_percentage)

# 设置字体
plt.rcParams['font.family'] = 'SimSun'
# 增大字体大小
plt.rcParams['font.size'] = 36

# 定义 100% 时的颜色
max_color = '#0d2d2a'

# 创建颜色映射
cmap = sns.light_palette(max_color, as_cmap=True)

# 指定最浅色对应的值
min_value = 10
max_value = 100

# 创建自定义的归一化对象
norm = Normalize(vmin=min_value, vmax=max_value)

# 调整图形尺寸，可根据实际情况调整
plt.figure(figsize=(10, 8))

# 绘制热图，同时调整注释字体大小和加粗，精细调整块之间的间距
# 调整颜色条与图的间距
sns.heatmap(df, annot=True, fmt=".1f", cmap=cmap, linewidths=5, annot_kws={"size": 36, "weight": "bold"},
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
plt.subplots_adjust(left=0.15, bottom=0.16, right=1, top=0.95)

# 移除大标题
# 保存图片，设置 dpi 提高清晰度
plt.savefig('./plotter/output/yolov8_single_data_kernels_accuracy_exact.png', dpi=100)
plt.close()
    