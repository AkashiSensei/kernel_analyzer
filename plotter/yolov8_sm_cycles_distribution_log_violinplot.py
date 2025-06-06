import json
import matplotlib.pyplot as plt
import numpy as np
import os
from typing import List, Dict, Any, Optional, Union
import math

# 设置中文字体为宋体
plt.rcParams["font.family"] = ["SimSun"]  # 仅使用宋体
plt.rcParams["axes.unicode_minus"] = False  # 解决负号显示问题

def read_json_data(file_path: str) -> Dict[str, List[float]]:
    """
    从JSON文件读取数据并转换为数值类型
    
    参数:
        file_path: JSON文件路径
        
    返回:
        包含数值数据的字典
    """
    try:
        with open(file_path, 'r') as f:
            data = json.load(f)
        
        # 转换所有数据为数值类型
        converted_data = {}
        for key, values in data.items():
            converted_values = []
            for value in values:
                try:
                    # 尝试将值转换为浮点数
                    converted_values.append(float(value))
                except (ValueError, TypeError) as e:
                    print(f"警告: 无法将值 '{value}' 转换为数值类型，已忽略")
            if converted_values:  # 只添加非空列表
                converted_data[key] = converted_values
        
        return converted_data
    except FileNotFoundError:
        print(f"错误: 文件 {file_path} 不存在")
        return {}
    except json.JSONDecodeError:
        print(f"错误: 文件 {file_path} 不是有效的JSON格式")
        return {}
    except Exception as e:
        print(f"错误: 读取文件 {file_path} 时发生异常: {e}")
        return {}

def apply_log_transform(data: Dict[str, List[float]], log_base: int = 10) -> Dict[str, List[float]]:
    """
    对数据应用对数转换
    
    参数:
        data: 原始数据字典
        log_base: 对数底数，默认为10
        
    返回:
        对数转换后的数据字典
    """
    transformed_data = {}
    for key, values in data.items():
        valid_values = []
        invalid_count = 0
        for value in values:
            if value > 0:  # 确保值大于0才能取对数
                valid_values.append(math.log(value, log_base))
            else:
                invalid_count += 1
        if invalid_count > 0:
            print(f"警告: 对 {key} 应用对数转换时，{invalid_count} 个值小于等于0被忽略")
        if valid_values:
            transformed_data[key] = valid_values
    return transformed_data

def plot_violin(data: Dict[str, List[float]], 
                output_path: Optional[str] = None,
                xlabel: str = "数据来源",
                ylabel: str = "数值",
                figsize: tuple = (10, 10),
                violin_color_list: Optional[List[str]] = None,
                scatter_color_list: Optional[List[str]] = None,
                show_points: bool = True,
                show_boxplot: bool = True,
                y_lim: Optional[tuple] = None,
                log_transform: bool = False,
                log_base: int = 10) -> None:
    """
    绘制小提琴图（支持独立的小提琴图和散点图颜色配置）
    
    参数:
        log_transform: 是否应用对数转换
        log_base: 对数转换的底数
    """
    if not data:
        print("错误: 没有数据可绘制")
        return
    
    # 应用对数转换
    if log_transform:
        data = apply_log_transform(data, log_base)
        # 更新y轴标签以反映对数转换
        ylabel = f"log{log_base}({ylabel})"
    
    plt.figure(figsize=figsize)
    labels = list(data.keys())
    values = list(data.values())
    
    # ===== 可调整参数区域 =====
    violin_width = 0.5  # 小提琴图宽度（0-1之间）
    violin_spacing = 0.6  # 小提琴图间距系数（值越大间距越大）
    violin_alpha = 0.5  # 小提琴图颜色透明度（0-1之间，越小越淡）
    scatter_alpha = 0.8  # 散点图颜色透明度（0-1之间）
    
    # 计算每个小提琴图的位置
    positions = np.arange(1, len(values) + 1) * violin_spacing
    
    # 绘制小提琴图
    violin_parts = plt.violinplot(values, positions=positions, widths=violin_width,
                                 showmeans=False, showmedians=True)
    
    # 应用小提琴图颜色
    if violin_color_list:
        for i, pc in enumerate(violin_parts['bodies']):
            violin_color = violin_color_list[i % len(violin_color_list)]
            pc.set_facecolor(violin_color)
            pc.set_edgecolor('white')
            pc.set_alpha(violin_alpha)
    else:
        for pc in violin_parts['bodies']:
            pc.set_alpha(violin_alpha)
    
    # 在小提琴图内添加箱线图
    if show_boxplot:
        boxprops = dict(
            linestyle='-', 
            color='black',
            linewidth=1.5,
            facecolor='#FFFFFF',
            alpha=0.8
        )
        whiskerprops = dict(linestyle='-', color='black', linewidth=1.5)
        medianprops = dict(linestyle='-', color='red', linewidth=2)
        plt.boxplot(values, widths=violin_width*0.3, positions=positions,
                   patch_artist=True, boxprops=boxprops, whiskerprops=whiskerprops,
                   medianprops=medianprops, showfliers=True, showmeans=False)
    
    # 添加散点显示原始数据点（使用独立的散点颜色列表）
    if show_points:
        if scatter_color_list:
            for i, (label, vals) in enumerate(zip(labels, values)):
                x = positions[i] + violin_width * 0.4 * (2 * (np.random.rand(len(vals)) - 0.5))
                scatter_color = scatter_color_list[i % len(scatter_color_list)]
                plt.scatter(x, vals, s=15, color=scatter_color, alpha=scatter_alpha)
        else:
            # 如果没有提供散点颜色列表，默认使用小提琴图颜色
            for i, (label, vals) in enumerate(zip(labels, values)):
                x = positions[i] + violin_width * 0.4 * (2 * (np.random.rand(len(vals)) - 0.5))
                color = violin_color_list[i % len(violin_color_list)] if violin_color_list else 'black'
                plt.scatter(x, vals, s=50, color=color, alpha=scatter_alpha)
    
    # 设置图表属性
    plt.xlabel(xlabel, fontsize=36, labelpad=5)
    plt.ylabel(ylabel, fontsize=36, labelpad=5)
    
    # 更新x轴刻度位置和标签
    plt.xticks(positions, labels, rotation=0, ha='center', fontsize=36)
    plt.yticks(fontsize=24)
    
    plt.tick_params(
        axis='both',
        which='both',
        pad=5,
    )
    
    # 调整图的位置和边距
    plt.subplots_adjust(
        left=0.18,
        bottom=0.25,
        right=0.95,
        top=0.95
    )
    
    # ===== 设置y轴范围 =====
    if y_lim is not None:
        if log_transform:
            # 如果是对数转换，将y_lim也转换为对数空间
            y_lim = (math.log(y_lim[0], log_base), math.log(y_lim[1], log_base))
        plt.ylim(y_lim)  # 设置指定的y轴范围
    
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    
    # 保存或显示图表
    if output_path:
        output_dir = os.path.dirname(output_path)
        if output_dir and not os.path.exists(output_dir):
            os.makedirs(output_dir)
        plt.savefig(output_path, dpi=100, bbox_inches='tight')
        print(f"图表已保存至: {output_path}")
    else:
        plt.show()

def main(json_file, output_file, ylabel, xlabel="YOLO计算图", y_lim=None, log_transform=False):
    """
    主函数，处理用户输入并绘制小提琴图
    """

    # 自定义颜色列表 - 程序会按顺序循环使用这些颜色
    violin_color_list = ["#e68b6f"] * 5
    scatter_color_list = ["#5c1011"] * 5
    
    data = read_json_data(json_file)
    if not data:
        return

    # 打印数据统计信息，帮助调试
    print("\n数据统计信息:")
    for key, values in data.items():
        if values:
            print(f"- {key}: {len(values)} 个样本, 范围: [{min(values)}, {max(values)}]")
        else:
            print(f"- {key}: 空数据")

    plot_violin(
        data=data,
        output_path=output_file,
        xlabel=xlabel,
        ylabel=ylabel,
        violin_color_list=violin_color_list,
        scatter_color_list=scatter_color_list,
        y_lim=y_lim,
        log_transform=log_transform
    )

if __name__ == "__main__":
    """
    Usage: python3 ./plotter/yolov8_sm_cycles_distribution_log_violinplot.py
    """

    # kernel 运行时数据分布
    main(
        "./results/ncu-value-extract/yolov8-sm-active-cycles.json",
        "./plotter/output/yolov8_sm_active_cycles_log_distribution_violinplot_limit.png",
        "SM 活跃周期数",
        log_transform=True  # 启用对数转换
    )    