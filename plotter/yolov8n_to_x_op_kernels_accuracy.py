import matplotlib.pyplot as plt
import numpy as np
from typing import Dict, List, Tuple, Optional

# 设置中文字体为宋体（确保中文正常显示）
plt.rcParams["font.family"] = ["SimSun"]  # 仅使用宋体
plt.rcParams["axes.unicode_minus"] = False  # 解决负号显示问题

def plot_accuracy_chart(
    data: Dict[str, Tuple[Tuple[int, int], Tuple[int, int]]],
    xlabel: str = "操作类型",
    ylabel: str = "准确率 (%)",
    output_path: Optional[str] = None,
) -> None:
    """
    绘制准确率堆积柱状图
    
    参数:
        data: 数据字典，格式为 {操作类型: ((精确准确数, 精确总数), (序列准确数, 序列总数))}
        xlabel: x轴标题
        ylabel: y轴标题
        output_path: 图片保存路径，None则显示图片
    """
    # ===== 可调整参数区域 =====
    FIGURE_SIZE = (20, 8)           # 图片大小
    BAR_WIDTH = 0.6                  # 柱状图宽度
    EXACT_COLOR = "#3D5855"          # 精确准确率颜色
    SEQUENCE_COLOR = "#BECBC9"       # 序列准确率颜色
    AXIS_LABEL_SIZE = 32             # 坐标轴标题字号
    TICK_LABEL_SIZE = 24             # 坐标轴刻度字号
    ANNOTATION_SIZE = 24             # 标注文字字号
    PADDING_TITLE = 5                # 标题与图的间距
    PADDING_TICK = 5                 # 刻度与图的间距
    DPI = 200                        # 图片分辨率
    ANNOTATION_OFFSET = 3.0          # 标注垂直偏移量(%)
    YLIM_PADDING = 5.0               # Y轴额外空间(%)
    # ===== 可调整参数区域结束 =====
    
    # 提取数据
    operations = list(data.keys())
    exact_counts = [data[op][0][0] for op in operations]
    exact_totals = [data[op][0][1] for op in operations]
    sequence_counts = [data[op][1][0] for op in operations]
    sequence_totals = [data[op][1][1] for op in operations]
    
    # 计算百分比
    exact_percentages = [count / total * 100 for count, total in zip(exact_counts, exact_totals)]
    sequence_percentages = [count / total * 100 for count, total in zip(sequence_counts, sequence_totals)]
    
    # 创建图表
    plt.figure(figsize=FIGURE_SIZE)
    
    # 绘制序列准确率柱状图
    bars_sequence = plt.bar(
        operations, 
        sequence_percentages, 
        width=BAR_WIDTH, 
        color=SEQUENCE_COLOR, 
        label='序列准确率'
    )
    
    # 绘制精确准确率柱状图
    bars_exact = plt.bar(
        operations, 
        exact_percentages, 
        width=BAR_WIDTH, 
        color=EXACT_COLOR, 
        label='精确准确率'
    )
    
    # 添加标注（原始数据）
    for i, (exact_pct, seq_pct, exact_data, seq_data) in enumerate(zip(
        exact_percentages, sequence_percentages, 
        zip(exact_counts, exact_totals),
        zip(sequence_counts, sequence_totals)
    )):
        # 精确准确率标注
        plt.text(
            i,
            exact_pct + ANNOTATION_OFFSET,  # 添加垂直偏移
            f"{exact_data[0]}/{exact_data[1]}",
            ha='center', 
            va='bottom',
            color='black',
            fontsize=ANNOTATION_SIZE,
            fontweight='bold'
        )
        
        # 序列准确率标注（仅当序列准确率高于精确准确率时）
        if seq_pct > exact_pct:
            plt.text(
                i,
                seq_pct + ANNOTATION_OFFSET,  # 添加垂直偏移
                f"{seq_data[0]}/{seq_data[1]}",
                ha='center', 
                va='bottom',
                color='black',
                fontsize=ANNOTATION_SIZE,
                fontweight='bold'
            )
    
    # 动态计算y轴上限，确保标注可见
    max_pct = max(max(exact_percentages), max(sequence_percentages))
    font_space = ANNOTATION_SIZE * 0.1  # 估计字体占用的空间
    ylim_upper = max(100, max_pct + ANNOTATION_OFFSET + font_space + YLIM_PADDING)
    plt.ylim(0, ylim_upper)
    
    # 设置坐标轴标签和刻度
    plt.xlabel(xlabel, fontsize=AXIS_LABEL_SIZE, labelpad=PADDING_TITLE)
    plt.ylabel(ylabel, fontsize=AXIS_LABEL_SIZE, labelpad=PADDING_TITLE)
    plt.xticks(rotation=45, ha='right', fontsize=TICK_LABEL_SIZE)
    plt.yticks(fontsize=TICK_LABEL_SIZE)
    
    # 调整刻度和坐标轴标题的距离
    plt.tick_params(axis='both', which='major', pad=PADDING_TICK)
    
    # 添加图例
    plt.legend(fontsize=TICK_LABEL_SIZE, loc='center right')
    
    # 调整图表位置和边距
    plt.subplots_adjust(
        left=0.12,
        bottom=0.20,
        right=0.95,
        top=0.95
    )
    
    # 保存或显示图表
    if output_path:
        plt.savefig(output_path, dpi=DPI, bbox_inches='tight')
        print(f"图表已保存至: {output_path}")
    else:
        plt.show()

def main():
    """示例：使用表格数据绘制准确率图表"""
    data = {
        "Conv": ((7, 104), (40, 104)),
        "Concat": ((6, 19), (19, 19)),
        "Split": ((1, 9), (9, 9)),
        "Slice": ((2, 2), (2, 2)),
        "Mul": ((36, 100), (100, 100)),
        "Add": ((3, 21), (21, 21)),
        "Sub": ((2, 2), (2, 2)),
        "Div": ((2, 2), (2, 2)),
        "Sigmoid": ((34, 98), (98, 98)),
        "MaxPool": ((0, 3), (3, 3)),
        "Softmax": ((1, 1), (1, 1)),
        "Transpose": ((1, 1), (1, 1)),
        "其它": ((9, 9), (9, 9)),
    }
    
    plot_accuracy_chart(
        data=data,
        xlabel="算子类型",
        ylabel="准确率 (%)",
        output_path="./plotter/output/yolov8n_to_x_op_kernels_accuracy_bar.png"
    )

if __name__ == "__main__":
    main()