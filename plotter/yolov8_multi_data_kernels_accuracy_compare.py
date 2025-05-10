import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from matplotlib.colors import Normalize, FuncNorm

exact_data = {'规则库数据来源计算图': ['V8n_s', 'V8s_l', 'V8l_x', 'V8m_l_x'], 'V8n': ['239/239', '146/239', '134/239', '134/239'], 'V8s': ['239/239', '239/239', '146/239', '144/239'], 'V8m': ['50/305', '51/305', '52/305', '305/305'], 'V8l': ['247/371', '371/371', '371/371', '371/371'], 'V8x': ['52/371', '37/371', '371/371', '371/371']}
seq_data = {'规则库数据来源计算图': ['V8n_s', 'V8s_l', 'V8l_x', 'V8m_l_x'], 'V8n': ['239/239', '223/239', '222/239', '222/239'], 'V8s': ['239/239', '239/239', '221/239', '223/239'], 'V8m': ['288/305', '290/305', '289/305', '305/305'], 'V8l': ['327/371', '371/371', '371/371', '371/371'], 'V8x': ['330/371', '359/371', '371/371', '371/371']}

def draw(data, output_path, start_value):
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
    plt.rcParams['font.size'] = 36

    # 定义 100% 时的颜色
    max_color = '#001a33'

    # 创建颜色映射
    cmap = sns.light_palette(max_color, as_cmap=True)

    # 指定最浅色对应的值
    min_value = start_value
    max_value = 100

    # 创建自定义的归一化对象
    norm = Normalize(vmin=min_value, vmax=max_value)

    # 调整图形尺寸，可根据实际情况调整
    plt.figure(figsize=(10, 7))

    # 绘制热图，同时调整注释字体大小和加粗，精细调整块之间的间距
    # 调整颜色条与图的间距
    sns.heatmap(df, annot=True, fmt=".1f", cmap=cmap, linewidths=5, annot_kws={"size": 36, "weight": "bold"},
                cbar_kws={"pad": 0.03}, norm=norm)

    # 添加横轴标题，并设置加粗，调整横轴标题和标签的距离
    plt.xlabel('被预测目标计算图', fontweight='bold', labelpad=15)
    # 设置纵轴标签并加粗，调整纵轴标题和标签的距离
    plt.ylabel('规则库数据来源计算图', fontweight='bold', labelpad=0)

    # 调整横轴刻度标签与图的间距
    plt.tick_params(axis='x', pad=10)
    # 调整纵轴刻度标签与图的间距
    plt.tick_params(axis='y', pad=10)

    # 将y轴刻度标签设置为水平显示
    plt.yticks(rotation=45,  ha='right', fontsize=28)

    # 调整子图布局，为标题和坐标轴留出更多空间
    plt.subplots_adjust(left=0.19, bottom=0.2, right=1, top=0.95)

    # 移除大标题
    # 保存图片，设置 dpi 提高清晰度
    plt.savefig(output_path, dpi=100)
    plt.close()

if __name__ == '__main__':
    """
    Usage: python3 ./plotter/yolov8_multi_data_kernels_accuracy_compare.py
    """
    draw(exact_data, './plotter/output/yolov8_multi_data_kernels_accuracy_exact.png', 10)
    draw(seq_data, './plotter/output/yolov8_multi_data_kernels_accuracy_seq.png', 85)
