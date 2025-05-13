import os
import pandas as pd
from pathlib import Path

def is_csv_valid(file_path):
    """检查CSV文件是否可以正常读取并包含特定数据"""
    try:
        # 读取前20行来检查文件结构
        with open(file_path, 'r', encoding='utf-8') as f:
            first_20_lines = [f.readline() for _ in range(20)]
        
        # 检查常见表头模式
        header_patterns = ['"ID"', '"Process ID"', 'ID,', 'Process ID,']
        has_header = any(pattern in line for line in first_20_lines for pattern in header_patterns)
        
        if not has_header:
            return False
            
        # 尝试读取CSV文件
        df = pd.read_csv(file_path)
        
        # 检查DataFrame是否有有效数据
        if len(df.columns) < 3 or df.empty:
            return False
            
        # 检查是否存在特定行：ID=1且Metric Name="Compute (SM) Throughput"
        target_row = df[(df['ID'] == 1) & (df['Metric Name'] == 'Compute (SM) Throughput')]
        
        if target_row.empty:
            print(f"警告: 文件 {file_path} 中未找到目标行")
            return False
            
        # 尝试获取Metric Value
        metric_value = target_row['Metric Value'].values[0]
        print(f"找到目标行，Metric Value: {metric_value}")
        return True
        
    except Exception as e:
        print(f"读取文件失败: {e}")
        return False

def remove_first_seven_lines(file_path):
    """删除文件的前7行"""
    with open(file_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    
    if len(lines) <= 7:
        print(f"文件 {file_path} 行数不足7行，跳过处理")
        return
    
    with open(file_path, 'w', encoding='utf-8') as f:
        f.writelines(lines[7:])
    print(f"已处理文件: {file_path}")

def main(directory):
    """主函数：处理目录下所有CSV文件"""
    directory_path = Path(directory)
    if not directory_path.exists() or not directory_path.is_dir():
        print(f"错误：目录 '{directory}' 不存在或不是有效目录")
        return
    
    # 获取目录下所有CSV文件
    csv_files = list(directory_path.glob('*.csv'))
    if not csv_files:
        print(f"目录 '{directory}' 下未找到CSV文件")
        return
    
    print(f"找到 {len(csv_files)} 个CSV文件")
    
    # 处理每个CSV文件
    for csv_file in csv_files:
        file_path = str(csv_file)
        print(f"正在检查文件: {file_path}")
        
        if is_csv_valid(file_path):
            print(f"文件格式正常，跳过: {file_path}")
            continue
        
        print(f"文件格式异常，尝试修复: {file_path}")
        remove_first_seven_lines(file_path)
        
        # 再次检查文件是否修复成功
        if is_csv_valid(file_path):
            print(f"修复成功: {file_path}")
        else:
            print(f"修复失败，文件仍无法正常读取: {file_path}")

if __name__ == "__main__":
    """
    Usage: python3 ./experiments/csv_processor.py
    """
    # 在这里设置要处理的目录路径
    target_directory = "./results/ncu/yolov8n-multi-runs"  # 修改为实际目录路径
    main(target_directory)    