import json
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
        # 处理不匹配的形状，这里仅会根据总大小给出一个基本的分数
        # wout.detailed(f"[data_based_kernel_finder]     形状不匹配，待测为 {len(testing_shape)}，目标为 {len(target_shape)}")
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
    return match_score

def _shape_full_score(shape_len, totol_size_weight):
    return shape_len + totol_size_weight

def _exact_output():
    print(f"[data_based_kernel_finder]     算子精确匹配")

def _no_exact_output(match_score, full_score=-1):
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

Returns:
    list: 找到的算子列表。空列表标识找到的结果就是空列表，即不真正调用 kernel；为 None 则表示异常或者没有找到。
"""
def empty_find_kernel(op_data_list, node_info):
    # 没有对应 kernel 的算子，直接返回空列表
    print(f"[data_based_kernel_finder]     {node_info['op_name']} 对应 kernel 为空")
    return []

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

    test_has_bias = "input_shape_2" in node_info

    difference_punish_weight = 1
    totol_size_weight = 2

    full_score = _shape_full_score(4, totol_size_weight) * 3
    if test_has_bias:
        # bias 是 1 维的
        full_score += _shape_full_score(1, totol_size_weight)
    target_data = {}
    max_exact_match_args_score = 0
    exact_match_args_score = 0

    for data in op_data_list:
        data_has_bias = "input_shape_2" in data
        
        if data_has_bias != test_has_bias:
            continue

        exact_match_args_score = 0
        exact_match_args_score += _calculate_shape_match_score(node_info["input_shape_0"], data["input_shape_0"], difference_punish_weight, totol_size_weight)
        exact_match_args_score += _calculate_shape_match_score(node_info["input_shape_1"], data["input_shape_1"], difference_punish_weight, totol_size_weight)
        if data_has_bias:
            exact_match_args_score += _calculate_shape_match_score(node_info["input_shape_2"], data["input_shape_2"], difference_punish_weight, totol_size_weight)
        exact_match_args_score += _calculate_shape_match_score(node_info["output_shape_0"], data["output_shape_0"], difference_punish_weight, totol_size_weight)

        if exact_match_args_score > max_exact_match_args_score:
            max_exact_match_args_score = exact_match_args_score
            target_data = data

            if full_score == exact_match_args_score:
                _exact_output()
                return target_data["kernels"]
    
    _no_exact_output(max_exact_match_args_score, full_score)

    print(target_data)
    
    # 当没有精确匹配时，调整最匹配序列的相关参数
    # 暂时先不做出调整，直接返回

    # 没有匹配的 kernel 序列，来源于是否有 input_2 
    if target_data == {}:
        wout.simple("[data_based_kernel_finder]     没有匹配的 kernel 序列，来源于是否有 input_2 的差异")
        return None

    return target_data["kernels"]

def concat_find_kernel(op_data_list, node_info):
    # 输入的数量不确定，不像 Conv 那样确定有图像、卷积核和偏置，该算子可能合并多个输入

    input_idx = 1  # 也表示输入的数量
    input_all_same = True
    input_dim = len(node_info["input_shape_0"])


    while f"input_shape_{input_idx}" in node_info:
        if node_info[f"input_shape_{input_idx}"] != node_info[f"input_shape_0"]:
            input_all_same = False
        input_idx += 1
    
    difference_punish_weight = 1
    totol_size_weight = 2

    full_score = _shape_full_score(input_dim, totol_size_weight) * (input_idx + 1)
    target_data = {}
    max_exact_match_args_score = 0
    exact_match_args_score = 0

    for data in op_data_list:
        # 不考虑输入数量不同的组合
        if f"input_shape_{input_idx}" in data or f"input_shape_{input_idx - 1}" not in data:
            continue

        # 区分考虑输入形状相同和不同的情况
        if input_all_same:
            # 输入形状完全一致
            if len(data["kernels"]) > 1:
                # 长度大于 1，代表存在内存操作，即使用的是通用 kernel 序列
                continue
        else:
            # 输入形状不完全一致
            if len(data["kernels"]) == 1:
                # 长度为 1，代表不存在内存操作，使用的是维度一致时的专用 kernel 序列
                continue
        
        exact_match_args_score = 0
        for idx in range(input_idx):
            exact_match_args_score += _calculate_shape_match_score(node_info[f"input_shape_{idx}"], data[f"input_shape_{idx}"], difference_punish_weight, totol_size_weight)
        exact_match_args_score += _calculate_shape_match_score(node_info["output_shape_0"], data["output_shape_0"], difference_punish_weight, totol_size_weight)

        if exact_match_args_score > max_exact_match_args_score:
            max_exact_match_args_score = exact_match_args_score
            target_data = data

            if full_score == exact_match_args_score:
                _exact_output()
                return target_data["kernels"]
            
    _no_exact_output(max_exact_match_args_score, full_score)
    
    if target_data == {}:
        wout.simple(f"[data_based_kernel_finder]     Concat 算子没有满足条件的 kernel 序列")
        return None

    # 当没有精确匹配时，调整最匹配序列的相关参数
    # 暂时先不做出调整，直接返回 
    return target_data["kernels"]
def split_find_kernel(op_data_list, node_info):
    # 类似 Concat，只不过数量不确定的是输出，可能分割为多个输出

    output_idx = 1  # 也表示输出的数量
    output_all_same = True
    output_dim = len(node_info["output_shape_0"])

    while f"output_shape_{output_idx}" in node_info:
        if node_info[f"output_shape_{output_idx}"] != node_info[f"output_shape_0"]:
            output_all_same = False
        output_idx += 1

    difference_punish_weight = 1
    totol_size_weight = 2

    full_score = _shape_full_score(output_dim, totol_size_weight) * (output_idx + 1)
    target_data = {}
    max_exact_match_args_score = 0
    exact_match_args_score = 0

    for data in op_data_list:
        # 不考虑输出数量不同的组合
        if f"output_shape_{output_idx}" in data or f"output_shape_{output_idx - 1}" not in data:
            continue

        # 区分考虑输出形状相同和不同的情况
        if output_all_same:
            # 输出形状完全一致
            if len(data["kernels"]) > 1:
                continue
        else:
            if len(data["kernels"]) == 2:
                continue

        exact_match_args_score = 0
        for idx in range(output_idx):
            exact_match_args_score += _calculate_shape_match_score(node_info[f"output_shape_{idx}"], data[f"output_shape_{idx}"], difference_punish_weight, totol_size_weight)
        exact_match_args_score += _calculate_shape_match_score(node_info["input_shape_0"], data["input_shape_0"], difference_punish_weight, totol_size_weight)

        if exact_match_args_score > max_exact_match_args_score:
            max_exact_match_args_score = exact_match_args_score
            target_data = data

            if full_score == exact_match_args_score:
                _exact_output()
                return target_data["kernels"]
            
    _no_exact_output(max_exact_match_args_score, full_score)

    if target_data == {}:
        wout.simple(f"[data_based_kernel_finder]     Split 算子没有满足条件的 kernel 序列")
        return None

    # 当没有精确匹配时，调整最匹配序列的相关参数
    # 暂时先不做出调整，直接返回 
    return target_data["kernels"]

def slice_find_kernel(op_data_list, node_info):
    # 目前见到的 Slice 包含 4 个输入，分别是 data, starts, ends, axes

    if "input_shape_3" not in node_info:
        wout.simple(f"[data_based_kernel_finder]     Slice 算子输入数量少于预期")
        return None
    
    difference_punish_weight = 1
    totol_size_weight = 2

    target_data = {}
    max_exact_match_args_score = 0
    exact_match_args_score = 0

    for data in op_data_list:
        exact_match_args_score = 0
        exact_match_args_score += _calculate_shape_match_score(node_info["input_shape_0"], data["input_shape_0"], difference_punish_weight, totol_size_weight)
        exact_match_args_score += _calculate_shape_match_score(node_info["input_shape_1"], data["input_shape_1"], difference_punish_weight, totol_size_weight)
        exact_match_args_score += _calculate_shape_match_score(node_info["input_shape_2"], data["input_shape_2"], difference_punish_weight, totol_size_weight)
        exact_match_args_score += _calculate_shape_match_score(node_info["input_shape_3"], data["input_shape_3"], difference_punish_weight, totol_size_weight)
        exact_match_args_score += _calculate_shape_match_score(node_info["output_shape_0"], data["output_shape_0"], difference_punish_weight, totol_size_weight)

        if exact_match_args_score > max_exact_match_args_score:
            max_exact_match_args_score = exact_match_args_score
            target_data = data
        
    return target_data["kernels"]

    
def simple_binary_find_kernel(op_data_list, node_info):
    # 逐元素操作，输入一定是两个，但是具体的维度不一定，会进行广播
    # 包含 Mul、Div、Add、Sub

    # 原始模型存在瑕疵，有 Div 节点常量形状参数未标注
    if len(node_info["input_shape_0"]) == 0:
        wout.simple("[data_based_kernel_finder]     空的输入形状 input_0，视为 [1]")
        node_info["input_shape_0"] = [1]
    if len(node_info["input_shape_1"]) == 0:
        wout.simple("[data_based_kernel_finder]     空的输入形状 input_1，视为 [1]")
        node_info["input_shape_1"] = [1]
        

    # 如果两个输入长度均为 1，则会被退回到 CPU 上
    if node_info["input_shape_0"] == [1] and node_info["input_shape_1"] == [1]:
        print(f"[data_based_kernel_finder]     两输入长度均为 1，认为算子将被退回到 CPU 上执行")
        return []
    
    difference_punish_weight = 1
    totol_size_weight = 2

    target_data = {}
    max_exact_match_args_score = 0
    exact_match_args_score = 0

    for data in op_data_list:
        # 原始模型存在瑕疵，有 Div 节点常量形状参数未标注
        if len(data["input_shape_0"]) == 0:
            data["input_shape_0"] = [1]
        if len(data["input_shape_1"]) == 0:
            data["input_shape_1"] = [1]

        exact_match_args_score = 0
        exact_match_args_score += _calculate_shape_match_score(node_info["input_shape_0"], data["input_shape_0"], difference_punish_weight, totol_size_weight)
        exact_match_args_score += _calculate_shape_match_score(node_info["input_shape_1"], data["input_shape_1"], difference_punish_weight, totol_size_weight)
        exact_match_args_score += _calculate_shape_match_score(node_info["output_shape_0"], data["output_shape_0"], difference_punish_weight, totol_size_weight)

        if exact_match_args_score > max_exact_match_args_score:
            max_exact_match_args_score = exact_match_args_score
            target_data = data

            if node_info["input_shape_0"] == data["input_shape_0"] and node_info["input_shape_1"] == data["input_shape_1"] and node_info["output_shape_0"] == data["output_shape_0"]:
                _exact_output()
                return data["kernels"]
    
    _no_exact_output(max_exact_match_args_score)

    return target_data["kernels"]

def simple_unary_find_kernel(op_data_list, node_info):
    # 输入和输出均只有一个
    input_dim = len(node_info["input_shape_0"])
    output_dim = len(node_info["output_shape_0"])
    
    difference_punish_weight = 1
    totol_size_weight = 2

    target_data = {}
    max_exact_match_args_score = 0
    exact_match_args_score = 0
    full_score = _shape_full_score(input_dim, totol_size_weight) + _shape_full_score(output_dim, totol_size_weight)

    for data in op_data_list:
        exact_match_args_score = _calculate_shape_match_score(node_info["input_shape_0"], data["input_shape_0"], difference_punish_weight, totol_size_weight)
        exact_match_args_score += _calculate_shape_match_score(node_info["output_shape_0"], data["output_shape_0"], difference_punish_weight, totol_size_weight)

        if exact_match_args_score > max_exact_match_args_score:
            target_data = data
            max_exact_match_args_score = exact_match_args_score

            if exact_match_args_score == full_score:
                _exact_output()
                return target_data["kernels"]
    
    _no_exact_output(max_exact_match_args_score, full_score)
    return target_data["kernels"]

def memory_find_kernel(op_data_list, node_info):
    # 内存拷贝节点暂略，因为这不在 ONNX 算子集中，而是 ORT 处理模型后在图中生成的
    wout.simple("[data_based_kernel_finder]     暂不支持内存拷贝节点，这并非 ONNX 算子集中节点")
    return None

op_func_dict = {
    "Conv": conv_find_kernel,
    "Concat": concat_find_kernel,
    "Split": split_find_kernel,
    "Slice": slice_find_kernel,

    "Mul": simple_binary_find_kernel,
    "Add": simple_binary_find_kernel,
    "Div": simple_binary_find_kernel,
    "Sub": simple_binary_find_kernel,

    "Sigmoid": simple_unary_find_kernel,
    "MaxPool": simple_unary_find_kernel,
    "Softmax": simple_unary_find_kernel,
    "Transpose": simple_unary_find_kernel,

    "Reshape": empty_find_kernel,
    "Resize": empty_find_kernel,
    "Shape": empty_find_kernel,
    "Gather": empty_find_kernel,

    "MemcpyFromHost": memory_find_kernel,
    "MemcpyToHost": memory_find_kernel,
}

def find_best_match_kernels(data, node_info):
    """
    寻找算子最为匹配的 kernel 序列。

    输入：
        `data` (dict): 数据收集器生成的数据，加载自文件。键为算子类型，如 “Conv” 。值为算子对应的 kernel 序列备选列表，列表中每个元素包含目标 kernel 序列和对应的算子参数，具体包括：
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

    Returns:
        list: 找到的算子列表。空列表标识找到的结果就是空列表，即不真正调用 kernel；为 None 则表示异常或者没有找到。
    """
    op_name = node_info["op_name"]
    print(f"[data_based_kernel_finder] 寻找 {node_info["op_name"]} {node_info["node_name"]} 算子匹配的 kernel 序列。")
    return op_func_dict.get(op_name, undefined_find_kernel)(data.get(op_name), node_info)
