import json
import sys
from utils import warning_output as wout
import math

def load_json_data(file_path):
    """
    从数据收集器保存的 Json 文件中读取数据，并返回，如果没有发现版本信息或者 data 字段，认为其并不是正确的数据，退出。

    Args:
        `file_path` (str): Json 文件路径。
    
    Returns:
        dict: 从文件中读取的数据，以算子名称为键的字典，值为列表，列表中每个元素包含目标 kernel 序列和对应的算子参数，具体包括：
            - "kernels": 该算子对应的 kernel 序列，是一个包含 kernel JSON 对象的列表。
            - "model": 模型名称。
            - "node_name": 节点名称。
            - "input_shape_{idx}": 各个输入形状，下标始于 0。
            - "output_shape_{idx}": 输出形状，下标始于 0。
    """
    with open(file_path, 'r') as f:
        data = json.load(f)

    if data is None:
        wout.error(f"[data_based_kernel_finder] Json 文件 {file_path} 读取时出错。")

    if "version" in data:
        print(f"[data_based_kernel_finder] 从文件 {file_path} 中读取数据, 版本 {data['version']}。")
    else:
        wout.error(f"[data_based_kernel_finder] Json 文件 {file_path} 中未找到版本信息，退出。", 2)
    
    if not "data" in data:
        wout.error(f"[data_based_kernel_finder] Json 文件 {file_path} 中未找到数据信息，退出。", 2)
    
    return data["data"]

"""
寻找匹配的辅助函数。
假定算子的输入已经经过计算。
"""

def _calculate_shape_match_score(testing_shape, target_shape, difference_punish_weight, totol_size_weight):
    """
    计算两形状匹配的得分。
    考量每个维度大小差异的同时，考虑整体尺寸

    Args:
        `testing_shape` (list): 待测形状。
        `target_shape` (list): 数据中形状。
        `difference_punish_weight` (float): 形状不同时，对形状差异的惩罚权重，需要小于 1。
        `totol_size_weight` (float): 形状大小权重。
    
    Returns:
        float: 得分
    """
    match_score = 0
    testing_size = 1
    target_size = 1

    if len(testing_shape) != len(target_shape):
        wout.detailed(f"[data_based_kernel_finder]     形状不匹配，待测为 {len(testing_shape)}，目标为 {len(target_shape)}")
        testing_size = math.prod(testing_shape)
        target_size = math.prod(target_shape)
        ratio = min(testing_size / target_size, target_size / testing_size) if testing_size != 0 and target_size != 0 else 0
        return totol_size_weight * ratio
    
    for testing, target in zip(testing_shape, target_shape):
        testing_size *= testing
        target_size *= target
        if testing == target:
            match_score += 1
        else:
            ratio = min(testing / target, target / testing) if testing != 0 and target != 0 else 0
            match_score += difference_punish_weight * ratio
    
    ratio = min(testing_size / target_size, target_size / testing_size) if testing_size != 0 and target_size != 0 else 0
    match_score += totol_size_weight * ratio

def _shape_full_score(shape_len, totol_size_weight):
    return shape_len + totol_size_weight

def _exact_output():
    print(f"[data_based_kernel_finder]     算子精确匹配")

def _no_exact_output(match_score, full_score):
    wout.simple(f"[data_based_kernel_finder]     算子未精确匹配，匹配度 {match_score} / {full_score}")

"""
寻找各个算子最为匹配的 kernel 序列。

输入：
    `op_data_list` (list): 算子对应的 kernel 序列备选，列表中每个元素包含目标 kernel 序列和对应的算子参数，具体包括：
                    - "kernels": 该算子对应的 kernel 序列，是一个包含 kernel JSON 对象的列表。
                    - "model": 模型名称。
                    - "node_name": 节点名称。
                    - "input_shape_{idx}": 各个输入形状，下标始于 0。
                    - "output_shape_{idx}": 输出形状，下标始于 0。
    `node_info` (dict): 输入算子信息，包含：
                    - "op_name": 算子名称。
                    - "node_name": 算子名称标识。
                    - "input_shape_{idx}": 各个输入形状，下标始于 0。
                    - "output_shape_{idx}": 输出形状，下标始于 0。
"""
def empty_find_kernel(op_data_list, node_info):
    # 没有对应 kernel 的算子，直接返回 None
    print(f"[data_based_kernel_finder]     {node_info['op_name']} 对应 kernel 为空")
    return None

def undefined_find_kernel(op_data_list, node_info):
    wout.simple(f"[data_based_kernel_finder]     {node_info['op_name']} 算子类型未注册")
    return None

def conv_find_kernel(op_data_list, node_info):
    # input_shape_0: input
    # input_shape_1: weight
    # input_shape_2: bias

    if len(node_info["input_shape_1"]) != 4:
        wout.simple(f"[data_based_kernel_finder]     Conv 算子权重 {node_info['input_shape_1']} 不是 4 维")
        return None

    difference_punish_weight = 1
    totol_size_weight = 2

    full_score = _shape_full_score(4, totol_size_weight) * 4
    target_kernels = []
    max_exact_match_args_score = 0
    exact_match_args_score = 0

    for data in op_data_list:
        exact_match_args_score = 0
        exact_match_args_score += _calculate_shape_match_score(node_info["input_shape_0"], data["input_shape_0"], difference_punish_weight, totol_size_weight)
        exact_match_args_score += _calculate_shape_match_score(node_info["input_shape_1"], data["input_shape_1"], difference_punish_weight, totol_size_weight)
        exact_match_args_score += _calculate_shape_match_score(node_info["input_shape_2"], data["input_shape_2"], difference_punish_weight, totol_size_weight)
        exact_match_args_score += _calculate_shape_match_score(node_info["output_shape_0"], data["output_shape_0"], difference_punish_weight, totol_size_weight)

        if exact_match_args_score > max_exact_match_args_score:
            max_exact_match_args_score = exact_match_args_score
            target_kernels = data["kernels"]

            if full_score == exact_match_args_score:
                _exact_output()
                return target_kernels
    
    _no_exact_output(exact_match_args_score, full_score)
    
    # 当没有精确匹配时，调整最匹配序列的相关参数
    # 暂时先不做出调整，直接返回
    return target_kernels

def concat_find_kernel(op_data_list, node_info):

    input_idx = 1  # 也表示输入的数量
    input_all_same = True
    input_dim = len(node_info["input_shape_0"])


    while f"input_shape_{input_idx}" in node_info:
        if node_info[f"input_shape_{input_idx}"] != node_info[f"input_shape_0"]:
            input_all_same = False
            break
        input_idx += 1
    
    difference_punish_weight = 1
    totol_size_weight = 2

    full_score = _shape_full_score(input_dim, totol_size_weight) * (input_idx + 1)
    target_kernels = []
    max_exact_match_args_score = 0
    exact_match_args_score = 0

    for data in op_data_list:
        # 不考虑输入数量不同的组合
        if f"input_shape_{input_idx}" in data or f"input_shape_{input_idx - 1}" not in data:
            continue

        # 区分考虑输入形状相同和不同的情况
        if input_all_same:
            # 输入形状完全一致
            if len(data["Kernels"]) > 1:
                # 长度大于 1，代表存在内存操作，即使用的是通用 kernel 序列
                continue
        else:
            # 输入形状不完全一致
            if len(data["Kernels"]) == 1:
                # 长度为 1，代表不存在内存操作，使用的是维度一致时的专用 kernel 序列
                continue
        
        exact_match_args_score = 0
        for idx in range(input_idx):
            exact_match_args_score += _calculate_shape_match_score(node_info[f"input_shape_{idx}"], data[f"input_shape_{idx}"], difference_punish_weight, totol_size_weight)
        exact_match_args_score += _calculate_shape_match_score(node_info["output_shape_0"], data["output_shape_0"], difference_punish_weight, totol_size_weight)

        if exact_match_args_score > max_exact_match_args_score:
            max_exact_match_args_score = exact_match_args_score
            target_kernels = data["kernels"]

            if full_score == exact_match_args_score:
                _exact_output()
                return target_kernels
            
    _no_exact_output(exact_match_args_score, full_score)
    
    if target_kernels == []:
        wout.simple(f"[data_based_kernel_finder]     Concat 算子没有满足条件的 kernel 序列")

    # 当没有精确匹配时，调整最匹配序列的相关参数
    # 暂时先不做出调整，直接返回 
    return target_kernels

def split_find_kernel(op_data_list, node_info):
    pass

op_func_dict = {
    "Conv": conv_find_kernel,
    "Concat": concat_find_kernel,

}

def find_best_match_kernels(data, node_info):
    op_name = node_info["op_name"]
    print(f"[data_based_kernel_finder] 寻找 {node_info["op_name"]} {node_info["node_name"]} 算子匹配的 kernel 序列。")
    return op_func_dict.get(op_name, undefined_find_kernel)(data.get(op_name), node_info)
    