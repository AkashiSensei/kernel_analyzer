import os
import subprocess
import time

"""
对指定的 Yolo 模型进行多次 Nsight Compute 分析，获取编号后的 csv 文件。

Usage:
    python3 ./profile/yolo_ncu_repeat.py
"""

# 指定模型路径和结果保存目录
# yolov8x
# model_path = './models/detection/ultralytics-yolov8/yolov8x.onnx'
# result_dir = './results/ncu/yolov8x-multi-runs'

# yolov8n
model_path = './models/detection/ultralytics-yolov8/yolov8n.onnx'
result_dir = './results/ncu/yolov8n-multi-runs'

# yolo12x
# model_path = './models/detection/ultralytics-yolov12/yolov12x.onnx'
# result_dir = './results/ncu/yolo12x-multi-runs'

# 运行次数
run_start = 31
num_runs = 20

os.makedirs(result_dir, exist_ok=True)

model_name = os.path.splitext(os.path.basename(model_path))[0]

for run_num in range(run_start, num_runs + run_start):
    result_file = os.path.join(result_dir, f'{model_name}-orto0-ncu-basic-run{run_num}.csv')

    print(f"发现 ONNX 模型文件: {os.path.basename(model_path)}")
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
        print(f'已完成对 {model_name} 的第 {run_num} 次性能分析，结果保存到 {result_file}')
        print(f"结束时间: {end_formatted_time}")
        print(f"耗时: {elapsed_time:.2f} 秒")
    except subprocess.CalledProcessError as e:
        end_time = time.time()  # 记录出错时的结束时间
        end_formatted_time = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(end_time))
        elapsed_time = end_time - start_time  # 计算耗时
        print(f'对 {model_name} 进行第 {run_num} 次性能分析时出错: {e}')
        print(f"结束时间: {end_formatted_time}")
        print(f"耗时: {elapsed_time:.2f} 秒")