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
        print(f"[mapping_relationship_finder] node_kernel_pairs 为空")
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

if __name__ == '__main__':
    node_kernel_pairs = tfp.get_pairs_from_trace_file("./examples/yolov8n-orto0.json")
    relation_2_instance_dict = get_relationship_dict(node_kernel_pairs)
    
    for relation, instance_dict in relation_2_instance_dict.items():
        print(f"Relation: {relation}")
        print(f"Count: {instance_dict['count']}")
        print()