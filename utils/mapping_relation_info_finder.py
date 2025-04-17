from collections import defaultdict
from utils import trace_file_parser as tfp

def get_relationship_dict(node_kernel_pairs):
    """
    生成关联关系的字典。字典的键是一个从算子到 kernel 的映射关系，值是该映射关系出现的次数，以及包含该映射关系的所有原始 `node_kernel_pairs` 实例列表。
    trace_kernel_reporter 中的分析与该函数类似，但有额外需要统计的信息，没有使用该函数

    Args:
        node_kernel_pairs: 一个列表，列表中的每个元素是一个字典，字典包含两个字段：
                        - "Node": 表示一个算子（Node）的 JSON 对象，其中包含算子的相关信息，如名称、参数大小、输入输出类型和形状等。
                        - "Kernels": 一个列表，包含该算子对应的所有 kernel 的 JSON 对象，这些 kernel 是按顺序排列的，包含 kernel 的名称、运行时长、网格和块大小等信息。

    Returns:
        dict: 关联字典，键为三元组 `(op_name, provider, kernel_sequence)`，其中：
              - `op_name` (str): 算子类型（如 "Add"、"Conv2D"）；
              - `provider` (str): 执行提供者（如 "CUDAExecutionProvider"）；
              - `kernel_sequence` (tuple): kernel 名称按执行顺序组成的元组。
              
              值为字典，包含：
              - `count` (int): 该关联关系的出现次数；
              - `instances` (list): 包含该关联关系的所有原始 `node_kernel_pairs` 实例列表。
    """
    if node_kernel_pairs is None:
        print(f"[mapping_relation_info_finder] node_kernel_pairs 为空")
        return None

    relation_2_instance_dict = defaultdict(lambda: {'count': 0, 'instances': []})

    for pair in node_kernel_pairs:
        current_node = pair["Node"]
        current_kernels = pair["Kernels"]

        op_name = current_node['args']['op_name']
        provider = current_node['args']['provider']
        kernel_sequence = [kernel['name'] for kernel in current_kernels]

        key = (op_name, provider, tuple(kernel_sequence))
        relation_2_instance_dict[key]['count'] += 1
        relation_2_instance_dict[key]['instances'].append(pair)

    return relation_2_instance_dict

def get_pairs_from_single_relation_dict(relation_2_instance_dict, relation_tuple):
    """
    生成关联关系的字典。字典的键是一个从算子到 kernel 的映射关系，值是该映射关系出现的次数，以及包含该映射关系的所有原始 `node_kernel_pairs` 实例列表。
    trace_kernel_reporter 中的分析与该函数类似，但有额外需要统计的信息，没有使用该函数

    Args:
        `relation_2_instance_dict` (dict): 关联字典，键为三元组 `(op_name, provider, kernel_sequence)`。值为字典，包含：
                                        - `count` (int): 该关联关系的出现次数；
                                        - `instances` (list): 包含该关联关系的所有原始 `node_kernel_pairs` 实例列表。
        
        `relation_tuple` (tuple): 三元组 `(op_name, provider, kernel_sequence)`，其中：
                      - `op_name` (str): 算子类型（如 "Add"、"Conv2D"）；
                      - `provider` (str): 执行提供者（如 "CUDAExecutionProvider"）；
                      - `kernel_sequence` (tuple): kernel 名称按执行顺序组成的元组。

    Returns:
        list: 一个列表，包含该映射关系的所有出现实例，列表中的每个元素是一个字典，字典包含两个字段：
            - "Node": 表示一个算子（Node）的 JSON 对象，其中包含算子的相关信息，如名称、参数大小、输入输出类型和形状等。
            - "Kernels": 一个列表，包含该算子对应的所有 kernel 的 JSON 对象，这些 kernel 是按顺序排列的，包含 kernel 的名称、运行时长、网格和块大小等信息。
    """
    if relation_2_instance_dict is None:
        # 字典生成出现问题
        print(f"[mapping_relation_info_finder] relation_2_instance_dict is None")
        return None
    
    instance_pairs = relation_2_instance_dict[relation_tuple]["instances"]

    if instance_pairs is None or len(instance_pairs) == 0:
        # 没有该映射关系的实例，元组长度为 1 时，不能省略最后的逗号
        print(f"[mapping_relation_info_finder] There is no instances for {relation_tuple}")
        return None

    assert(len(instance_pairs) != 0)

    return instance_pairs


def get_args_from_pairs(instance_pairs, kernel_idx):
    """
    从 `instance_pairs` 中提取参数信息，并生成一个包含这些参数的列表。

    Args:
        `instance_pairs` (list): 一个列表，包含该映射关系的所有出现实例，列表中的每个元素是一个字典，字典包含两个字段：
            - "Node": 表示一个算子（Node）的 JSON 对象，其中包含算子的相关信息，如名称、参数大小、输入输出类型和形状等。
            - "Kernels": 一个列表，包含该算子对应的所有 kernel 的 JSON 对象，这些 kernel 是按顺序排列的，包含 kernel 的名称、运行时长、网格和块大小等信息。
        `kernel_idx` (int): 需要提取的 kernel 的索引。

    Returns:
        list: 一个列表，包含从 `instance_pairs` 中提取的参数信息，列表中的每个元素是一个字典，字典包含两个字段：
            - "op_args": 包含算子的参数信息，如输入输出类型和形状、参数大小、激活大小、运行时长等。
            - "kernel_args": 包含 kernel 的参数信息，如网格大小、块大小、线程大小、运行时长等。
    """
    args_pairs = []

    for instance_pair in instance_pairs:
        op = instance_pair["Node"]
        kernel = instance_pair["Kernels"][kernel_idx]

        new_args_pair = {
            "op_args":{
                # "input_type_shape": op["args"]["input_type_shape"], # 列表，观察效果欠佳
                # "output_type_shape": op["args"]["output_type_shape"], # 列表，观察效果欠佳
                "output_size": int(op["args"]["output_size"]),
                "parameter_size": int(op["args"]["parameter_size"]),
                # "activation_size": int(op["args"]["activation_size"]), # 实测数据，ONNX 图节点中没有
                # "duration": op["dur"], # 实测数据，ONNX 图节点中没有
            },
            "kernel_args": {
                "grid_x": int(kernel["args"]["grid_x"]),
                "grid_y": int(kernel["args"]["grid_y"]),
                "grid_z": int(kernel["args"]["grid_z"]),
                "block_x": int(kernel["args"]["block_x"]),
                "block_y": int(kernel["args"]["block_y"]),
                "block_z": int(kernel["args"]["block_z"]),
                "duration": kernel["dur"],
            }
        }
        for idx, input in enumerate(op["args"]["input_type_shape"]):
            input_shape = list(input.values())[0]
            new_args_pair["op_args"][f"input_shape_{idx}"] = input_shape

        for idx, output in enumerate(op["args"]["output_type_shape"]):
            output_shape = list(output.values())[0]
            new_args_pair["op_args"][f"output_shape_{idx}"] = output_shape

        new_args_pair["kernel_args"]["grid_size"] = new_args_pair["kernel_args"]["grid_x"] * new_args_pair["kernel_args"]["grid_y"] * new_args_pair["kernel_args"]["grid_z"]
        new_args_pair["kernel_args"]["block_size"] = new_args_pair["kernel_args"]["block_x"] * new_args_pair["kernel_args"]["block_y"] * new_args_pair["kernel_args"]["block_z"]
        new_args_pair["kernel_args"]["thread_size"] = new_args_pair["kernel_args"]["grid_size"] * new_args_pair["kernel_args"]["block_size"]

        args_pairs.append(new_args_pair)

    return args_pairs

if __name__ == '__main__':
    node_kernel_pairs = tfp.get_pairs_from_trace_file("./examples/yolov8n-orto0.json")
    relation_2_instance_dict = get_relationship_dict(node_kernel_pairs)
    
    for relation, instance_dict in relation_2_instance_dict.items():
        print(f"Relation: {relation}")
        print(f"Count: {instance_dict['count']}")
        print()