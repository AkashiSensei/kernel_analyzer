import pandas as pd
import os
import json
from typing import List, Dict, Optional, Union
from utils import warning_output as wout

def extract_metric_from_csv(
    file_path: str, 
    metric_name: str,
    target_column_name: str,
    metric_column: str = 'Metric Name',
    value_column: str = 'Metric Value'
) -> pd.DataFrame:
    """
    从单个CSV文件中提取指定指标的数据
    
    Args:
        file_path: CSV文件路径
        metric_name: 要提取的指标名称
        target_column_name: 提取后的数据列名
        metric_column: 指标名称所在的列名
        value_column: 指标值所在的列名
        
    Returns:
        提取并处理后的DataFrame
    """
    try:
        # 读取CSV文件
        df = pd.read_csv(file_path)
        
        # 筛选指定指标的行
        filtered_df = df[df[metric_column] == metric_name].copy()
        
        # 提取值列并重命名
        if not filtered_df.empty:
            extracted = filtered_df[value_column].rename(target_column_name)
            # 去除空值
            cleaned = extracted.dropna()
            return cleaned.to_frame()
        else:
            print(f"警告: 文件 {file_path} 中未找到指标 '{metric_name}'")
            return pd.DataFrame()
    except Exception as e:
        print(f"错误: 处理文件 {file_path} 时发生异常: {e}")
        return pd.DataFrame()

def save_to_json(data: Dict[str, List[Union[int, float]]], output_path: str) -> None:
    """
    将提取的数据保存为JSON格式
    
    参数:
        data: 要保存的数据，格式为 {列名: [值1, 值2, ...]}
        output_path: 输出JSON文件路径
    """
    output_dir = os.path.dirname(output_path)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    with open(output_path, 'w') as f:
        json.dump(data, f, indent=2)
    print(f"成功保存整合数据到: {output_path}")

def main(
    metric_name: str,
    file_configs: List[Dict[str, str]],
    output_path: str,
    metric_column: str = 'Metric Name',
    value_column: str = 'Metric Value',
    fill_value: Optional[Union[str, int, float]] = None
) -> None:
    """
    主函数，执行完整的数据提取和合并流程
    
    参数:
        metric_name: 要提取的指标名称
        file_configs: 配置列表，每个配置包含文件路径和目标列名
        output_path: 输出文件路径（支持CSV或JSON格式，根据扩展名自动判断）
        metric_column: 指标名称所在的列名
        value_column: 指标值所在的列名
        fill_value: 用于填充缺失值的值，仅在保存为CSV时有效
    """
    # 提取数据
    extracted_data = {}
    for config in file_configs:
        file_path = config['file_path']
        column_name = config['column_name']
        df = extract_metric_from_csv(
            file_path, 
            metric_name, 
            column_name,
            metric_column=metric_column,
            value_column=value_column
        )
        if not df.empty:
            # 转换为列表保存（保留原始索引信息）
            extracted_data[column_name] = df[column_name].tolist()
    
    # 根据输出文件扩展名选择保存方式
    if not extracted_data:
        wout.error("没有提取到任何数据，无法生成输出文件")
        return
    
    output_ext = os.path.splitext(output_path)[1].lower()
    if output_ext == '.json':
        save_to_json(extracted_data, output_path)
    elif output_ext == '.csv':
        dfs = [pd.DataFrame(data=values, columns=[col_name]) 
               for col_name, values in extracted_data.items()]
        # 使用outer join保留所有数据点
        if dfs:
            merged_df = dfs[0]
            for df in dfs[1:]:
                merged_df = merged_df.merge(df, left_index=True, right_index=True, how='outer')
            if fill_value is not None:
                merged_df = merged_df.fillna(fill_value)
            merged_df.to_csv(output_path, index=False)
            print(f"成功保存整合数据到: {output_path}")
    else:
        wout.error(f"不支持的输出格式: {output_ext}，请使用.csv或.json")

if __name__ == "__main__":
    """
    Usage:
        python3 ./experiments/csv_metric_value_extractor.py
    """
    # 基本参数
    METRIC_COLUMN = 'Metric Name'  # 指标名称所在的列名
    VALUE_COLUMN = 'Metric Value'   # 指标值所在的列名
    FILL_VALUE = None               # 填充缺失值的值，仅在保存为CSV时有效



    # YOLOV8 模型上的 Grid Size、Block Size、Registers Per Thread
    # METRIC_NAME = "Registers Per Thread"  # 要提取的指标名称
    # MODEL_IDENTIFIERS = ["n", "s", "m", "l", "x"]
    # FILE_CONFIGS = [
    #     {
    #         "file_path": f"./results/ncu/ultralytics-yolov8/yolov8{identifier}-orto0-ncu-basic.csv",
    #         "column_name": f"V8{identifier}"
    #     } for identifier in MODEL_IDENTIFIERS
    # ]
    # OUTPUT_PATH = "results/ncu-value-extract/yolov8-registers-per-thread.json"  # 输出文件路径（修改为.json以使用JSON格式）


    # YOLOV8 模型上的 Compute (SM) Throughput、Memory Throughput、SM Active Cycles
    METRIC_NAME = "SM Active Cycles"  # 要提取的指标名称
    MODEL_IDENTIFIERS = ["n", "s", "m", "l", "x"]
    FILE_CONFIGS = [
        {
            "file_path": f"./results/ncu/ultralytics-yolov8/yolov8{identifier}-orto0-ncu-basic.csv",
            "column_name": f"V8{identifier}"
        } for identifier in MODEL_IDENTIFIERS
    ]
    OUTPUT_PATH = "results/ncu-value-extract/yolov8-sm-active-cycles.json"  # 输出文件路径（修改为.json以使用JSON格式）




    # 执行主函数
    main(
        metric_name=METRIC_NAME,
        file_configs=FILE_CONFIGS,
        output_path=OUTPUT_PATH,
        metric_column=METRIC_COLUMN,
        value_column=VALUE_COLUMN,
        fill_value=FILL_VALUE
    )