import matplotlib.pyplot as plt
import numpy as np
import os

# 可配置部分，方便统一修改
data = [
    [
        [32, 8, 7, 5, 4, 4, 2, 1, 1],
        [32, 9, 8, 4, 4, 2, 1, 1, 1, 1, 1],
        [22, 8, 7, 6, 4, 4, 2, 2, 2, 1, 1, 1, 1, 1, 1, 1]
    ],
    [
        [32, 8, 7, 5, 4, 4, 2, 1, 1],
        [32, 9, 8, 4, 4, 2, 1, 1, 1, 1, 1],
        [21, 14, 9, 6, 3, 2, 2, 2, 1, 1, 1, 1, 1]
    ],
    [
        [32, 24, 7, 1],
        [32, 24, 7, 1],
        [32, 24, 7, 1],
    ]
]

# 直接指定每组的颜色列表
color_groups = [
    ['#aef8f9', '#aed3f9', '#aef9d5', '#f9afae'],
    ['#a5caee', '#a5a5ee', '#a5eeed', '#eec9a5'],
    ['#f1bbc1', '#f1d0bb', '#f1bbdc', '#bbf1eb']
]

# 分别指定每个系列的名称
series_names = ['Tesla-V100-SXM2-32GB', 'Tesla-T4', 'NVIDIA-A100-SXM4-40GB']

# 设置 x 轴标签
x_axis_labels = ['EXHAUSTIVE', 'HEURISTIC', 'DEFAULT']
# x 轴标题
x_axis_title = 'Conv 算法搜索策略'
# y 轴标题
y_axis_title = '对应到某类 kernel 序列的算子数量'
# 图例标题
legend_title = '堆积类别'

# 绘图相关参数
bar_width = 0.2
# 每组有 3 个柱子
num_bars_per_group = 3
# 总共有 3 组
num_groups = 3
# 中文字体
font_family = 'SimSun'
# 图片分辨率
dpi = 200
# 保存图片的文件名
# output_filename = '1.png'
current_filename = os.path.basename(__file__)
output_filename = os.path.splitext(current_filename)[0] + '.png'

# 指定中文字体为宋体
plt.rcParams['font.family'] = font_family

# 计算 x 轴的位置
x = np.arange(num_groups)

# 循环绘制每组柱子
for i in range(num_bars_per_group):
    bottoms = np.zeros(num_groups)
    color_list = color_groups[i]
    num_colors = len(color_list)
    for j in range(max([len(data[k][i]) for k in range(num_groups)])):
        values = []
        for k in range(num_groups):
            if j < len(data[k][i]):
                values.append(data[k][i][j])
            else:
                values.append(0)
        # 根据索引循环使用颜色
        color_index = j % num_colors
        color = color_list[color_index]
        plt.bar(x + i * bar_width, values, width=bar_width, bottom=bottoms,
                color=color)
        bottoms += np.array(values)

    # 在柱子顶端显示堆积数量
    for k in range(num_groups):
        total_height = bottoms[k]
        num_stacks = len(data[k][i])
        plt.text(x[k] + i * bar_width, total_height, str(num_stacks), ha='center', va='bottom')

# 设置 x 轴标签
plt.xticks(x + bar_width * (num_bars_per_group - 1) / 2, x_axis_labels)

# 添加 x 轴和 y 轴标题
plt.xlabel(x_axis_title)
plt.ylabel(y_axis_title)

# 生成图例
legend_handles = []
for i, color_list in enumerate(color_groups):
    # 取每个颜色组中间的颜色（如果颜色组长度为奇数）
    middle_index = len(color_list) // 2
    middle_color = color_list[middle_index]
    handle = plt.Rectangle((0, 0), 1, 1, color=middle_color)
    legend_handles.append(handle)

# 添加图例
plt.legend(legend_handles, series_names, title=legend_title)

# 保存图片并指定分辨率（dpi）
plt.savefig(output_filename, dpi=dpi, bbox_inches='tight')
# plt.show()  # 若不需要展示，可注释掉这行