import os
import subprocess

"""
对 Yolo 模型进行批量 Nsight Compute 分析，获取 csv 文件。

Usage:
    python3 ./profile/yolo_ncu_csv.py
"""

model_dir = './models/detection/ultralytics-yolov8'
result_dir = './results/ncu/ultralytics-yolov8'
os.makedirs(result_dir, exist_ok=True)

for filename in os.listdir(model_dir):
    if filename.endswith('.onnx'):
        model_name = os.path.splitext(filename)[0]
        result_file = os.path.join(result_dir, f'{model_name}-orto0-ncu-basic.csv')
        model_path = os.path.join(model_dir, filename)

        print(f"发现 ONNX 模型文件: {filename}")
        print(f"模型路径: {model_path}")
        print(f"CSV 保存路径: {result_file}")
        print("开始进行性能分析...")

        command = f'ncu --csv > {result_file} python3 ./inference/detection/yolo_ort_nopf.py --model {model_path} >/dev/null 2>&1'

        try:
            subprocess.run(command, shell=True, check=True)
            print(f'已完成对 {model_name} 的性能分析，结果保存到 {result_file}')
        except subprocess.CalledProcessError as e:
            print(f'对 {model_name} 进行性能分析时出错: {e}')
    