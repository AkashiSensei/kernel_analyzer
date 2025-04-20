from collections import defaultdict
import json
import os
from utils import trace_file_parser as tfp
import sys
"""
用于生成基于规则的分析模型使用的数据，目前仅仅基于跟踪文件，目标为 kernel 序列和其 block 和 grid 数，
后续可能考虑结合 Nsys 或者 Ncu 的数据，进行对应，让 kernel 数据更为完整和丰富。
版本 v1.0。
"""
def build_data_from_single_trace_file(trace_file_path, batch_size=1, data=None):
    """
    基于传入的数据，在其之上补充来自跟踪文件的数据。

    Args:
        `trace_file_path` (str): 跟踪文件路径。
        `data` (dict): 数据字典，默认为 None，非空时将基于它进一步补充数据，用于创建来自于多个跟踪文件的数据。
    
    Returns:
        dict: 以算子名称为键的字典，值为列表，列表中每个元素包含目标 kernel 序列和对应的算子参数，具体包括：
            - "kernels": 该算子对应的 kernel 序列，是一个包含 kernel JSON 对象的列表。
            - "model": 模型名称。
            - "node_name": 节点名称。
            - "input_shape_{idx}": 各个输入形状，下标始于 0。
            - "output_shape_{idx}": 输出形状，下标始于 0。
    """
    node_kernel_pairs = tfp.get_pairs_from_trace_file(trace_file_path)
    op_name_2_pairs_dict = tfp.divide_pairs_by_op_name(node_kernel_pairs)

    if op_name_2_pairs_dict is None or len(op_name_2_pairs_dict) == 0:
        print("[model_data_collector] No pairs found from trace file {trace_file_path}.")
        sys.exit(1)
    
    model_name = os.path.splitext(os.path.basename(trace_file_path))[0]
    print(f"[model_data_collector] Build data from trace file {trace_file_path} of model {model_name}.")

    if data is None:
        data = defaultdict(list)

    for op_name, pairs in op_name_2_pairs_dict.items():
        print(f"[model_data_collector] Processing operator: {op_name}")

        for pair in pairs:
            data_entry = {
                "kernels" : pair["Kernels"],
                "model" : model_name,
                "node_name": pair["Node"]["name"],
                "batch_size": batch_size,
            }
            for idx, input in enumerate(pair["Node"]["args"]["input_type_shape"]):
                input_shape = list(input.values())[0]
                data_entry[f"input_shape_{idx}"] = input_shape

            for idx, output in enumerate(pair["Node"]["args"]["output_type_shape"]):
                output_shape = list(output.values())[0]
                data_entry[f"output_shape_{idx}"] = output_shape

            data[op_name].append(data_entry)
            
    return data

def save_data_to_json(output_file_path, data, gpu, description, version):
    """
    将数据保存到 json 文件中。

    Args:
        `output_file_path` (str): 保存路径。
        `data` (dict): 要保存的数据，以算子名称为键的字典，值为列表，列表中每个元素包含目标 kernel 序列和对应的算子参数，具体包括：
                    - "kernels": 该算子对应的 kernel 序列，是一个包含 kernel JSON 对象的列表。
                    - "model": 模型名称。
                    - "node_name": 节点名称。
                    - "input_shape_{idx}": 各个输入形状，下标始于 0。
                    - "output_shape_{idx}": 输出形状，下标始于 0。
        `gpu` (str): GPU 名称。
        `version` (str): 数据版本。
    """
    data_save = {
        "gpu": gpu,
        "version": version,
        "data": data,
        "description": description,
    }

    with open(output_file_path, "w") as f:
        json.dump(data_save, f, indent=4)

    print(f"[model_data_collector] Data saved to {output_file_path}.")

if __name__ == "__main__":
    """
    usage: python3 ./rule_based_model/model_data_collector.py
    """

    # 构建基于 yolov8n，关闭优化的数据
    data = build_data_from_single_trace_file("./examples/yolov8n-orto0.json")
    save_data_to_json(
        "./rule_based_model/data/yolov8n-orto0.json", 
        data, 
        "Tesla V100-SXM2-32GB",
        "data of yolov8n.onnx with optimization off, batch size 1.", 
        "1.0"
    )