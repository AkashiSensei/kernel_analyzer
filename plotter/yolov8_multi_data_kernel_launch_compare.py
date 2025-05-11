import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.patches import Patch

# 设置中文字体支持
plt.rcParams['font.family'] = 'SimSun'
plt.rcParams["axes.unicode_minus"] = False  # 正确显示负号

def create_gradient_bar(ax, x_pos, height, width, cmap, num_segments=50):
    """创建左右方向渐变色的柱状图"""
    for i in range(num_segments):
        segment_width = width / num_segments
        segment_x = x_pos + i * segment_width
        color = cmap(i / (num_segments - 1))
        ax.bar(segment_x, height, segment_width, color=color, edgecolor='none')

def plot_prediction_accuracy(data_table, x_label, y_label, save_path, color_list, 
                             legend_ncol=3, y_min=0, y_max=105, tick_cnt=11):
    """
    绘制预测准确率对比图，将两个模型的柱子合并在一个图表中
    
    参数:
    - data_table: 包含模型准确率的数据字典
    - x_label: x轴标签
    - y_label: y轴标签
    - save_path: 图片保存路径
    - color_list: 颜色列表，每组两个颜色表示渐变色
    - legend_ncol: 图例列数
    - y_min, y_max: y轴范围
    - inner_gap: 组内柱子间距
    """
    # ====== 可调整的绘图参数 ======
    fig_width, fig_height = 12, 8  # 图片大小
    fig_left, fig_bottom, fig_width_ratio, fig_height_ratio = 0.1, 0.1, 0.85, 0.85  # 图表位置
    
    # 坐标轴参数
    axis_label_fontsize = 40      # 坐标轴标题字号
    tick_label_fontsize = 32      # 坐标轴刻度字号
    title_pad = 10                # 坐标轴标题与图的间距
    tick_pad = 5                  # 坐标轴刻度与图的间距
    
    # 柱状图参数 - 重新设计间距
    bar_width = 0.5              # 柱子宽度
    gap_between_bars = 0.8       # 组内柱子间距（柱子中心到中心的距离）
    gap_between_models = 1.5     # 不同模型组之间的间距
    opacity = 0.85               # 透明度
    
    # 网格和边框
    show_grid = True              # 是否显示网格
    grid_alpha = 0.3              # 网格透明度
    spine_width = 1.2             # 边框粗细
    
    # 图例
    legend_fontsize = 28          # 图例字号
    legend_location = 'center right'  # 图例位置固定
    legend_bbox = (0.98, 0.4)     # 图例位置调整
    legend_ncol = 1               # 设置为单列
    legend_handle_length = 1.5    # 图例中色块的长度
    legend_column_spacing = 0.8   # 图例列之间的间距
    legend_handle_text_spacing = 0.5  # 图例色块与文本之间的间距
    legend_borderpad = 0.3        # 图例内边距
    
    # 标题和文本
    show_values_on_bars = True    # 是否在柱子上显示数值
    value_fontsize = 24           # 数值字号
    value_format = '{:.1f}'      # 数值格式
    
    # ============================
    
    # 提取数据
    models = data_table['规则库数据来源计算图']
    model_types = list(data_table.keys())[1:]  # 排除第一个键(数据源名称)
    
    # 确保有两个模型类型
    if len(model_types) != 2:
        raise ValueError("需要两个模型类型")
    
    # 创建画布
    fig = plt.figure(figsize=(fig_width, fig_height))
    ax = fig.add_axes([fig_left, fig_bottom, fig_width_ratio, fig_height_ratio])
    
    # 设置x轴位置 - 将两个模型组分开
    group_width = (len(models) - 1) * gap_between_bars + bar_width  # 一组柱子的总宽度
    group_positions = [0, group_width + gap_between_models]
    
    # 计算每组柱子的中心位置
    group1_center = group_positions[0] + group_width / 2
    group2_center = group_positions[1] + group_width / 2
    
    # 绘制柱状图
    for i, model_type in enumerate(model_types):
        for j, model in enumerate(models):
            accuracy = data_table[model_type][j] * 100  # 转换为百分比
            
            # 创建渐变色映射
            cmap = LinearSegmentedColormap.from_list(
                f'custom_{i}_{j}', [color_list[j][0], color_list[j][1]])
            
            # 计算柱子位置（考虑组内间距）
            x_pos = group_positions[i] + j * gap_between_bars
            
            # 绘制渐变色柱子（左右方向）
            create_gradient_bar(ax, x_pos, accuracy, bar_width, cmap)
            
            # 在柱子上方显示数值（居中对齐）
            if show_values_on_bars:
                ax.text(x_pos + bar_width/2, accuracy + 0.05, value_format.format(accuracy),
                        ha='center', va='bottom', fontsize=value_fontsize)
    
    # 设置x轴标签和刻度
    ax.set_xlabel(x_label, fontsize=axis_label_fontsize, labelpad=title_pad)
    ax.set_ylabel(y_label, fontsize=axis_label_fontsize, labelpad=title_pad)

    # 设置刻度标签与图的间距
    ax.tick_params(axis='both', which='major', pad=tick_pad)
    
    # 设置x轴刻度和标签（使用计算出的组中心位置）
    ax.set_xticks([group1_center, group2_center])
    ax.set_xticklabels(model_types, fontsize=tick_label_fontsize, rotation=0)
    
    # 设置x轴范围，确保两组柱子都能显示完整
    total_width = group_positions[1] + group_width
    ax.set_xlim(-0.5, total_width + 0.5)
    
    # 设置y轴范围和刻度
    ax.set_ylim(y_min, y_max)
    y_ticks = np.linspace(y_min, y_max, tick_cnt)  # 生成11个均匀分布的刻度
    ax.set_yticks(y_ticks)
    ax.set_yticklabels([f'{tick:.0f}%' for tick in y_ticks], fontsize=tick_label_fontsize)
    
    # 添加网格线
    if show_grid:
        ax.grid(axis='y', alpha=grid_alpha)
    
    # 设置边框粗细
    for spine in ax.spines.values():
        spine.set_linewidth(spine_width)
    
    # 创建图例
    # 创建图例 - 使用渐变的中间颜色
    def get_mid_color(color1, color2):
        """计算两个颜色的中间值"""
        from matplotlib.colors import to_rgb
        r1, g1, b1 = to_rgb(color1)
        r2, g2, b2 = to_rgb(color2)
        return f'#{int((r1+r2)/2*255):02x}{int((g1+g2)/2*255):02x}{int((b1+b2)/2*255):02x}'
    
    legend_items = [Patch(facecolor=get_mid_color(color_list[j][0], color_list[j][1]), 
                          label=models[j]) for j in range(len(models))]
    
    ax.legend(
        handles=legend_items,
        fontsize=legend_fontsize,
        loc=legend_location,
        bbox_to_anchor=legend_bbox,
        ncol=legend_ncol,
        handlelength=legend_handle_length,
        columnspacing=legend_column_spacing,
        handletextpad=legend_handle_text_spacing,
        borderpad=legend_borderpad,  # 设置图例内边距
        borderaxespad=0.0  # 减少图例与图表的间距
    )

    # 调整布局并保存（移除可能导致警告的tight_layout）
    plt.savefig(save_path, dpi=200, bbox_inches='tight')
    plt.close()

if __name__ == '__main__':
    """
    Usage: python3 ./plotter/yolov8_multi_data_kernel_launch_compare.py
    """
    n_color = '#8cd4cc'
    m_color = '#4db8ff'
    x_color = '#c78cd7'

    # 定义颜色列表，每组两个颜色表示渐变色
    color_list = [
        [n_color, n_color],  # V8n
        [m_color, m_color],  # V8m
        [x_color, x_color],  # V8x
        [n_color, m_color],  # V8n_m
        [n_color, x_color],  # V8n_x

        [m_color, x_color],  # V8m_x
    ]

    # 数据
    grid_size_table = {
        '规则库数据来源计算图': ['V8n', 'V8m', 'V8x', 'V8n_m', 'V8n_x', 'V8m_x'],
        'YOLOV8s': [0.7580802344826915, 0.8171054672992265, 0.7532653203058741, 
            0.858972697656989, 0.8290203192611514, 0.8246302173429707],
        'YOLOV8l': [0.579132221720108, 0.8201502330889074, 0.806329835531815, 
            0.8303190392145611, 0.8304489871962782, 0.8334908533836838]
    }

    block_size_table = {
        '规则库数据来源计算图': ['V8n', 'V8m', 'V8x', 'V8n_m', 'V8n_x', 'V8m_x'],
        'YOLOV8s': [0.9403068340306834, 0.9826850156903766, 0.9758716875871686, 
            0.9601360922245467, 0.9494421199442118, 0.9826850156903766],
        'YOLOV8l': [0.9307213451418309, 0.9915094339622641, 0.995575022461815, 
            0.9921832884097034, 0.995575022461815, 0.9942273135669363]
    }

    register_size_table = {
        '规则库数据来源计算图': ['V8n', 'V8m', 'V8x', 'V8n_m', 'V8n_x', 'V8m_x'],
        'YOLOV8s': [0.9625791261101032, 0.9872315781742314, 0.9751081815873138, 
            0.9717803984922632, 0.9676567515820763, 0.9872315781742314],
        'YOLOV8l': [0.9480771150227145, 0.9892648602729189, 0.9959842742128295, 
            0.9893611251939818, 0.9959842742128295, 0.9935354292335147]
    }

    # 绘制并保存图表
    plot_prediction_accuracy(grid_size_table, 
                            '被测计算图', 
                            'Grid大小预测准确率(%)', 
                            './plotter/output/yolov8_multi_grid_size_accuracy.png', 
                            color_list,
                            legend_ncol=3,
                            y_min=50, y_max=90)

    plot_prediction_accuracy(block_size_table, 
                            '被测计算图', 
                            'Block大小预测准确率(%)', 
                            './plotter/output/yolov8_multi_block_size_accuracy.png', 
                            color_list,
                            legend_ncol=3,
                            y_min=93, y_max=100,
                            tick_cnt=8)

    plot_prediction_accuracy(register_size_table, 
                            '被测计算图', 
                            '寄存器数预测准确率(%)', 
                            './plotter/output/yolov8_multi_register_accuracy.png', 
                            color_list,
                            legend_ncol=3,
                            y_min=94, y_max=100,
                            tick_cnt=7)

    print("图表已成功保存！")            