import os
import subprocess
import time

"""
对 Yolo 模型进行批量 Nsight Compute 分析，获取 csv 文件。

Usage:
    python3 ./profile/yolo_ncu_csv.py
"""

# Yolov8 model profile
# model_dir = './models/detection/ultralytics-yolov8'
# result_dir = './results/ncu/ultralytics-yolov8'

# Yolo11 model profile
# model_dir = './models/detection/ultralytics-yolo11'
# result_dir = './results/ncu/ultralytics-yolo11'

# Yolo12 model profile
# model_dir = './models/detection/ultralytics-yolo12'
# result_dir = './results/ncu/ultralytics-yolo12'


# Yolov8 model profile retest
# model_dir = './models/detection/ultralytics-yolov8'
# result_dir = './results/ncu/retest-yolov8'

# Yolo11 model profile
model_dir = './models/detection/ultralytics-yolo11'
result_dir = './results/ncu/retest-yolo11'

# Yolo12 model profile
# model_dir = './models/detection/ultralytics-yolo12'
# result_dir = './results/ncu/retest-yolo12'

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

        start_time = time.time()  # 记录开始时间
        start_formatted_time = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(start_time))
        print(f"开始时间: {start_formatted_time}")

        command = f'ncu --csv >{result_file} python3 ./inference/detection/yolo_ort_nopf.py --model {model_path}'

        try:
            subprocess.run(command, shell=True, check=True)
            end_time = time.time()  # 记录结束时间
            end_formatted_time = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(end_time))
            elapsed_time = end_time - start_time  # 计算耗时
            print(f'已完成对 {model_name} 的性能分析，结果保存到 {result_file}')
            print(f"结束时间: {end_formatted_time}")
            print(f"耗时: {elapsed_time:.2f} 秒")
        except subprocess.CalledProcessError as e:
            end_time = time.time()  # 记录出错时的结束时间
            end_formatted_time = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(end_time))
            elapsed_time = end_time - start_time  # 计算耗时
            print(f'对 {model_name} 进行性能分析时出错: {e}')
            print(f"结束时间: {end_formatted_time}")
            print(f"耗时: {elapsed_time:.2f} 秒")
    