import json
"""
该模块包含两个函数，用于解析 ONNX Profiler 跟踪文件并生成算子与 kernel 的对应关系。
仅适用于开启 GPU Profiling 的 ONNX Profiler 跟踪文件，且模型必须以默认的串行方式执行。
"""

def get_pairs_from_trace_file(trace_file_path):
    """
    从指定的 ONNX Profiler 跟踪文件中解析出每个算子（Node）及其对应的 kernel 序列。

    Args:
        trace_file_path (str): 包含 ONNX Profiler 输出的跟踪文件的路径，该文件为 JSON 格式。

    Returns:
        list: 一个列表，列表中的每个元素是一个字典，字典包含两个字段：
            - "Node": 表示一个算子（Node）的 JSON 对象，其中包含算子的相关信息，如名称、参数大小、输入输出类型和形状等。
            - "Kernels": 一个列表，包含该算子对应的所有 kernel 的 JSON 对象，这些 kernel 是按顺序排列的，包含 kernel 的名称、运行时长、网格和块大小等信息。

    Raises:
        FileNotFoundError: 如果指定的文件路径不存在，会打印错误信息。
        json.JSONDecodeError: 如果文件无法解析为有效的 JSON 格式，会打印错误信息。
        Exception: 如果发生其他未知错误，会打印相应的错误信息。
    """
    try:
        with open(trace_file_path, 'r') as file:
            data = json.load(file)

        node_kernel_pairs = []
        current_node = None
        current_kernels = []
        node_idx = 1
        kernel_idx = 1

        for item in data:
            if item["cat"] == "Node":
                node_idx += 1
                if current_node is not None:
                    node_kernel_pairs.append({"Node": current_node, "Kernels": current_kernels})
                current_node = item
                current_node["Index"] = node_idx
                current_kernels = []
            elif item["cat"] == "Kernel":
                kernel_idx += 1
                if current_node is not None:
                    item["Index"] = kernel_idx
                    current_kernels.append(item)

        if current_node is not None:
            node_kernel_pairs.append({"Node": current_node, "Kernels": current_kernels})

        return node_kernel_pairs
    except FileNotFoundError:
        print(f"[trace_file_parser] 错误：文件 {trace_file_path} 未找到。")
    except json.JSONDecodeError:
        print(f"[trace_file_parser] 错误：无法解析 {trace_file_path} 为有效的 JSON 文件。")
    except Exception as e:
        print(f"[trace_file_parser] 发生未知错误：{e}")


def get_node_kernel_mapping(node_kernel_pairs):
    """
    根据算子与kernel的对应关系列表，生成通过node的name找到对应的kernel序列、算子本身以及该组映射在列表中的序号（从1开始）的字典。

    Args:
        node_kernel_pairs (list): 该列表由get_pairs_from_trace_file函数返回，列表中的每个元素是一个字典，包含两个字段：
                                - "Node": 表示一个算子（Node）的JSON对象，其中包含算子的相关信息，如名称、参数大小、输入输出类型和形状等。
                                - "Kernels": 一个列表，包含该算子对应的所有kernel的JSON对象，这些kernel是按顺序排列的，包含kernel的名称、运行时长、网格和块大小等信息。

    Returns:
        dict: 生成的字典，键为算子（Node）的name，值为另一个字典，包含以下字段：
            - "Node": 算子本身的JSON对象，包含算子的相关信息，如名称、参数大小、输入输出类型和形状等。
            - "Kernels": 该算子对应的kernel序列，是一个包含kernel JSON对象的列表。
    """
    mapping_dict = {}
    for index, pair in enumerate(node_kernel_pairs):
        node_name = pair["Node"]["name"]
        mapping_dict[node_name] = {
            "Node": pair["Node"],
            "Kernels": pair["Kernels"],
        }
    return mapping_dict


if __name__ == "__main__":
    # 测试解析函数
    file_path = "./examples/yolov8n-orto0.json"
    parsed_result = get_pairs_from_trace_file(file_path)
    if parsed_result:
        for item in parsed_result:
            print("Node:", item["Node"]["name"], "Index:", item["Node"]["Index"])
            print("Kernels:")
            for kernel in item["Kernels"]:
                print(f"    {kernel['Index']}:{kernel['name']}")
            print()

    # 生成映射字典并测试（yolov8n 模型的数据）
    mapping_dict = get_node_kernel_mapping(parsed_result)
    for key in mapping_dict:
        print(key)
    print()
    pair_info = mapping_dict.get("/model.22/dfl/conv/Conv_kernel_time")
    if pair_info:
        print("Node:", pair_info["Node"]["name"], "Index:", pair_info["Node"]["Index"])
        print("Kernels:")
        for kernel in pair_info["Kernels"]:
            print(f"    {kernel['Index']}:{kernel['name']}")
    else:
        print("[trace_file_parser] 未找到对应的Node")