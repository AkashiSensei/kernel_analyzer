from collections import defaultdict
import json
import os
import pandas as pd
from pathlib import Path
from utils import warning_output as wout

def get_kernel_metric_value(id_df, metric_name):
    target_row = id_df[id_df['Metric Name'] == metric_name]

    if len(target_row) == 0:
        wout.error(f"[kernel_execute_metric_fluctuate_analysis] Metric {metric_name} not found in the dataframe.")
    if len(target_row) > 1:
        wout.error(f"[kernel_execute_metric_fluctuate_analysis] Multiple rows found for metric {metric_name}.")

    return target_row.iloc[0]['Metric Value']

def get_kernels_metrics_table(kernel_id_list, file_dir, metric_list):
    # 遍历目录下所有 csv 文件
    kernel_metrics_tables = defaultdict(lambda: defaultdict(list))

    for file_name in os.listdir(file_dir):
        if file_name.endswith('.csv'):
            file_path = os.path.join(file_dir, file_name)
            # 读取 csv 文件
            df = pd.read_csv(file_path)

            for kernel_id in kernel_id_list:
                # 获取 kernel_id 对应的行
                id_df = df[df['ID'] == kernel_id]
                if id_df.empty:
                    wout.error(f"[kernel_execute_metric_fluctuate_analysis] No data found for kernel ID {kernel_id} in file {file_path}.")
                
                # 获取指定指标的值
                for metric_name in metric_list:
                    metric_value = get_kernel_metric_value(id_df, metric_name)
                    kernel_metrics_tables[metric_name][kernel_id].append(metric_value)
    
    return kernel_metrics_tables

def save_table_to_json_file(table, output_path):
    with open(output_path, 'w') as f:
        json.dump(table, f, indent=4)
    print(f"[kernel_execute_metric_fluctuate_analysis] Table saved to {output_path}")

                
def main():
    kernel_id_list = [0, 3, 303, 326]
    metric_list = ["Compute (SM) Throughput", "Memory Throughput", "SM Active Cycles", "SM Frequency", "Duration", "Achieved Occupancy"]
    csv_file_dir = './results/ncu/yolov8n-multi-runs'
    output_json_path = './results/fluctuate/yolov8n_50times_kernels0_3_303_326.json'

    kernel_metrics_tables = get_kernels_metrics_table(kernel_id_list, csv_file_dir, metric_list)
    save_table_to_json_file(kernel_metrics_tables, output_json_path)

if __name__ == '__main__':
    """
    Usage: python3 ./experiments/kernel_execute_metric_fluctuate_analysis.py
    """
    main()