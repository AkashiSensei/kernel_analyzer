import json
import matplotlib.pyplot as plt
import numpy as np
import os
from typing import List, Dict, Any, Optional, Union

# 设置中文字体为宋体
plt.rcParams["font.family"] = ["SimSun"]  # 仅使用宋体
plt.rcParams["axes.unicode_minus"] = False  # 解决负号显示问题

def read_json_data(file_path: str) -> Dict[str, Dict[str, List[float]]]:
    """
    从JSON文件读取嵌套数据结构：指标 -> Kernel ID -> 数据列表
    """
    try:
        with open(file_path, 'r') as f:
            data = json.load(f)
        
        converted_data = {}
        for metric, kernel_data in data.items():
            converted_kernel = {}
            for kernel_id, values in kernel_data.items():
                try:
                    converted_values = [float(v) for v in values]
                    if converted_values:
                        converted_kernel[kernel_id] = converted_values
                except (ValueError, TypeError) as e:
                    print(f"警告: 处理 {metric} 的 {kernel_id} 数据时出错: {e}")
            if converted_kernel:
                converted_data[metric] = converted_kernel
        
        return converted_data
    except FileNotFoundError:
        print(f"错误: 文件 {file_path} 不存在")
        return {}
    except json.JSONDecodeError:
        print(f"错误: 文件 {file_path} 不是有效的JSON格式")
        return {}
    except Exception as e:
        print(f"错误: 读取文件时发生异常: {e}")
        return {}

def normalize_data(kernel_data: Dict[str, List[float]]) -> tuple:
    """
    对每个Kernel的数据进行归一化（以各自平均值为基准）
    返回: (归一化后数据, 基准值字典)
    """
    normalized = {}
    baselines = {}
    for kernel_id, values in kernel_data.items():
        baseline = np.mean(values)  # 以平均值作为基准值
        normalized_values = [(v / baseline) * 100 for v in values]
        normalized[kernel_id] = normalized_values
        baselines[kernel_id] = f"{baseline:.2f}"  # 保留两位小数
    return normalized, baselines

def plot_violin_for_metric(
    metric_data: Dict[str, List[float]],
    output_path: str,
    xlabel: str = "Kernel ID",
    ylabel: str = "归一化值（%）",
    figsize: tuple = (8, 10),
    violin_color_list: Optional[List[str]] = None,
    scatter_color_list: Optional[List[str]] = None,
    show_points: bool = True,
    show_boxplot: bool = True,
    unit: str = "",
    y_lim: Optional[tuple] = None
) -> None:
    """
    为单个指标绘制小提琴图（包含数据归一化和基准值标注）
    """
    if not metric_data:
        print("错误: 指标数据为空")
        return
    
    # 数据归一化处理
    normalized_data, baselines = normalize_data(metric_data)
    kernel_ids = list(normalized_data.keys())
    values = list(normalized_data.values())
    
    plt.figure(figsize=figsize)
    positions = np.arange(1, len(values) + 1)
    violin_width = 0.6
    violin_spacing = 1.0
    
    # 绘制小提琴图
    violin_parts = plt.violinplot(
        values, positions=positions, widths=violin_width,
        showmeans=False, showmedians=True
    )
    
    # 应用颜色配置
    default_colors = ["#b5b9c8", "#6c7bc4", "#4e9a43", "#d46c9b"]
    violin_colors = violin_color_list or default_colors
    scatter_colors = scatter_color_list or default_colors
    
    for i, pc in enumerate(violin_parts['bodies']):
        pc.set_facecolor(violin_colors[i % len(violin_colors)])
        pc.set_edgecolor('white')
        pc.set_alpha(0.6)
    
    # 添加箱线图
    if show_boxplot:
        plt.boxplot(
            values, positions=positions, widths=violin_width*0.4,
            patch_artist=True, boxprops={'facecolor': 'white', 'edgecolor': 'black', 'alpha': 0.8},
            whiskerprops={'color': 'black'}, medianprops={'color': 'red'}
        )
    
    # 添加散点图
    if show_points:
        for i, (kernel_id, vals) in enumerate(zip(kernel_ids, values)):
            x_jitter = positions[i] + (np.random.rand(len(vals)) - 0.5) * violin_width*0.5
            plt.scatter(
                x_jitter, vals, s=20, 
                color=scatter_colors[i % len(scatter_colors)], alpha=0.8
            )
    
    # 设置图表属性
    plt.xlabel(xlabel, fontsize=36, labelpad=28)
    plt.ylabel(ylabel, fontsize=36, labelpad=10)
    plt.xticks(positions, kernel_ids, rotation=0, ha='center', fontsize=36)
    plt.yticks(fontsize=28)
    
    # 智能调整底部边距和标注位置
    if y_lim:
        y_min, y_max = y_lim
        # 计算标注位置（在y轴范围下方5%处）
        label_position = y_min - (y_max - y_min) * 0.064
        # 调整底部边距，基于图表内容高度动态调整
        bottom_margin = 0.2  # 默认值
        if (y_max - y_min) < 20:  # 如果y轴范围小于20%，适当减小底部边距
            bottom_margin = 0.15
    else:
        # 如果没有指定y_lim，使用原来的逻辑
        label_position = -5
        bottom_margin = 0.3
    
    # 添加基准值标注
    for i, kernel_id in enumerate(kernel_ids):
        baseline = baselines[kernel_id]
        baseline = float(baseline)
        if baseline > 1000000000:
            baseline = f"{baseline / 1000000000:.2f}G"
        elif unit == "μs":
            baseline = f"{baseline / 1000:.1f}"
        elif unit == "":
            baseline = f"{baseline:.0f}"
        plt.text(
            positions[i], label_position,
            f"{baseline}{unit}",
            ha='center', va='top',
            fontsize=32,
            color='#1c2e39'
        )
    
    # 设置y轴范围
    if y_lim:
        plt.ylim(y_lim)
    else:
        # 如果没有指定y_lim，确保标注可见
        plt.ylim(bottom=label_position - 5)
    
    plt.grid(axis='y', linestyle='--', alpha=0.5)
    plt.subplots_adjust(left=0.15, bottom=bottom_margin, right=0.95, top=0.95)
    
    # 保存图表
    output_dir = os.path.dirname(output_path)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)
    plt.savefig(output_path, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"已保存 {ylabel} 图表到 {output_path}")

def main(
    json_file: str,
    output_paths: List[str],
    ylabels: List[str],
    xlabel: str = "Kernel ID",
    violin_color_list: Optional[List[str]] = None,
    scatter_color_list: Optional[List[str]] = None,
    y_lim: Optional[tuple] = None,
    unit_list: list = []
) -> None:
    """
    主函数，处理多指标绘图并校验路径一致性
    """
    full_data = read_json_data(json_file)
    metrics = list(full_data.keys())
    
    if len(output_paths) != len(metrics) or len(ylabels) != len(metrics):
        raise ValueError("输出路径列表或标签列表长度必须与指标数量一致")
    
    for metric, output_path, ylabel, unit in zip(metrics, output_paths, ylabels, unit_list):
        metric_data = full_data[metric]
        plot_violin_for_metric(
            metric_data=metric_data,
            output_path=output_path,
            xlabel=xlabel,
            ylabel=ylabel,
            violin_color_list=violin_color_list,
            scatter_color_list=scatter_color_list,
            y_lim=y_lim,
            unit=unit
        )

if __name__ == "__main__":
    """
    Usage: python3 ./plotter/yolov8_kernel_fluctuate_violinplot.py
    """

    main(
        json_file="./results/fluctuate/yolov8n_50times_kernels0_3_303_326.json",
        output_paths=[
            "./plotter/output/yolov8n_4kernels_fluctuate_c_throughput.png",
            "./plotter/output/yolov8n_4kernels_fluctuate_m_throughput.png",
            "./plotter/output/yolov8n_4kernels_fluctuate_sm_active_cycles.png",
            "./plotter/output/yolov8n_4kernels_fluctuate_sm_frequency.png",
            "./plotter/output/yolov8n_4kernels_fluctuate_duration.png",
            "./plotter/output/yolov8n_4kernels_fluctuate_achieved_occupancy.png"
        ],
        ylabels=[
            "计算吞吐利用率（%）",
            "存储吞吐利用率（%）",
            "SM 活跃周期数（%）",
            "SM 频率（%）",
            "持续时间（%）",
            "达到的占用率（%）"
        ],
        violin_color_list=["#b5b9c8"] * 4,
        scatter_color_list=["#2e3e4a"] * 4,
        y_lim=(92.5, 107.5),
        unit_list=["%", "%", "", "Hz", "μs", "%"]
    )