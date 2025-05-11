from collections import defaultdict
import json
import os
from utils import trace_file_parser as tfp
import sys
from utils import warning_output as wout
from utils import pairs_ncu_integrator as pni
"""
用于生成基于规则的分析模型使用的数据，目前仅仅基于跟踪文件，目标为 kernel 序列和其 block 和 grid 数，
后续可能考虑结合 Nsys 或者 Ncu 的数据，进行对应，让 kernel 数据更为完整和丰富。
版本 v2.0。

v2.0:
    增加了对于来自 ncu 的 csv 数据的支持。

"""
def build_data_from_single_trace_file(trace_file_path, batch_size=1, ncu_csv_path=None, data=None):
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
    if ncu_csv_path is not None:
        node_kernel_pairs = pni.fill_pairs_with_ncu(node_kernel_pairs, ncu_csv_path)
    op_name_2_pairs_dict = tfp.divide_pairs_by_op_name(node_kernel_pairs)

    if op_name_2_pairs_dict is None or len(op_name_2_pairs_dict) == 0:
        wout.error("[model_data_collector] No pairs found from trace file {trace_file_path}.")
    
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
                    - "kernels": 该算子对应的 kernel 序列，是一个包含 kernel JSON 对象的列表，注意 K 是小写。
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
    # data = build_data_from_single_trace_file("./examples/yolov8n-orto0.json")
    # save_data_to_json(
    #     "./rule_based_model/data/yolov8n-orto0-origin.json", 
    #     data, 
    #     "Tesla V100-SXM2-32GB",
    #     "data of yolov8n.onnx with optimization off, batch size 1.", 
    #     "1.0"
    # )

    # 构建基于 yolov8 不同版本的单一模型，关闭优化，包含 ncu kernel 详细分析数据的数据
    # data = build_data_from_single_trace_file("./results/trace/yolov8-orto0/yolov8n-orto0.json", ncu_csv_path="./results/ncu/ultralytics-yolov8/yolov8n-orto0-ncu-basic.csv")
    # save_data_to_json(
    #     "./rule_based_model/data/single-yolov8/yolov8n-orto0-ncu.json", 
    #     data, 
    #     "Tesla V100-SXM2-32GB",
    #     "data of yolov8n.onnx with optimization off, batch size 1, filled with basic data from ncu.", 
    #     "2.0"
    # )

    # 构建基于 yolov8 不同版本的符合模型，关闭优化，包含 ncu kernel 详细分析数据的数据，用于 kernel 预测准确率分析
    # 具体包括 n_s s_l l_x m_l_x
    # data = build_data_from_single_trace_file("./results/trace/yolov8-orto0/yolov8m-orto0.json", ncu_csv_path="./results/ncu/ultralytics-yolov8/yolov8m-orto0-ncu-basic.csv")
    # data = build_data_from_single_trace_file("./results/trace/yolov8-orto0/yolov8l-orto0.json", ncu_csv_path="./results/ncu/ultralytics-yolov8/yolov8l-orto0-ncu-basic.csv", data=data)
    # data = build_data_from_single_trace_file("./results/trace/yolov8-orto0/yolov8x-orto0.json", ncu_csv_path="./results/ncu/ultralytics-yolov8/yolov8x-orto0-ncu-basic.csv", data=data)
    # save_data_to_json(
    #     "./rule_based_model/data/multi-yolov8/yolov8m_l_x-orto0-ncu.json", 
    #     data, 
    #     "Tesla V100-SXM2-32GB",
    #     "data of yolov8m, yolov8l & yolov8x with optimization off, batch size 1, filled with basic data from ncu.", 
    #     "2.0"
    # )

    # 构建基于 yolov8 不同版本的符合模型，关闭优化，包含 ncu kernel 详细分析数据的数据，用于 kernel 启动数据预测准确率分析
    # 具体包括 n_m n_x m_x ，测试其在另外两个模型上的表现
    data = build_data_from_single_trace_file("./results/trace/yolov8-orto0/yolov8n-orto0.json", ncu_csv_path="./results/ncu/ultralytics-yolov8/yolov8n-orto0-ncu-basic.csv")
    data = build_data_from_single_trace_file("./results/trace/yolov8-orto0/yolov8m-orto0.json", ncu_csv_path="./results/ncu/ultralytics-yolov8/yolov8m-orto0-ncu-basic.csv", data=data)
    save_data_to_json(
        "./rule_based_model/data/multi-yolov8/yolov8n_m-orto0-ncu.json", 
        data, 
        "Tesla V100-SXM2-32GB",
        "data of yolov8n & yolov8m with optimization off, batch size 1, filled with basic data from ncu.", 
        "2.0"
    )
