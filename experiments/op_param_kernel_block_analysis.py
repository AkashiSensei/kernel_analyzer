from utils import trace_file_parser as tfp
from utils import mapping_relationship_finder as mrf

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
        print(f"[op_param_kernel_block_analysis] relation_2_instance_dict is None")
        return None
    
    instance_pairs = relation_2_instance_dict[relation_tuple]["instances"]

    if instance_pairs is None:
        print(f"[op_param_kernel_block_analysis] There is no instances for {relation_tuple}")
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
                "input_type_shape": op["args"]["input_type_shape"],
                "output_type_shape": op["args"]["output_type_shape"],
                "output_size": int(op["args"]["output_size"]),
                "parameter_size": int(op["args"]["parameter_size"]),
                # "activation_size": int(op["args"]["activation_size"]),
                # "duration": op["dur"],
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

        new_args_pair["kernel_args"]["grid_size"] = new_args_pair["kernel_args"]["grid_x"] * new_args_pair["kernel_args"]["grid_y"] * new_args_pair["kernel_args"]["grid_z"]
        new_args_pair["kernel_args"]["block_size"] = new_args_pair["kernel_args"]["block_x"] * new_args_pair["kernel_args"]["block_y"] * new_args_pair["kernel_args"]["block_z"]
        new_args_pair["kernel_args"]["thread_size"] = new_args_pair["kernel_args"]["grid_size"] * new_args_pair["kernel_args"]["block_size"]

        args_pairs.append(new_args_pair)

    return args_pairs


def console_report(relation_2_instance_dict, relation_tuple, kernel_idx):
    """
    打印 `relation_2_instance_dict` 中指定 `relation_tuple` 和 `kernel_idx` 的参数信息。

    Args:
        `relation_2_instance_dict` (dict): 关联字典，键为三元组 `(op_name, provider, kernel_sequence)`。值为字典，包含：
                                        - `count` (int): 该关联关系的出现次数；
                                        - `instances` (list): 包含该关联关系的所有原始 `node_kernel_pairs` 实例列表。
        `relation_tuple` (tuple): 三元组 `(op_name, provider, kernel_sequence)`，其中：
                      - `op_name` (str): 算子类型（如 "Add"、"Conv2D"）；
                      - `provider` (str): 执行提供者（如 "CUDAExecutionProvider"）；
                      - `kernel_sequence` (tuple): kernel 名称按执行顺序组成的元组。
        `kernel_idx` (int): 需要提取的 kernel 的索引。
    """

    instance_pairs = get_pairs_from_single_relation_dict(relation_2_instance_dict, relation_tuple)
    args_pairs = get_args_from_pairs(instance_pairs, kernel_idx)

    print(f"Op: {relation_tuple[0]}")
    print(f"EP: {relation_tuple[1]}")
    print(f"Kernels: {relation_tuple[2]}")
    print(f"Kernel idx: {kernel_idx}")
    print(f"Kernel name: {relation_tuple[2][kernel_idx]}")
    print()
    print()

    # 按照 `output_size` 排序
    args_pairs.sort(key=lambda x: x["op_args"]["output_size"])

    for idx, args_pair in enumerate(args_pairs):
        print(f"Instance {idx + 1}/{len(args_pairs)}")
        print("  op_args:")
        for key, value in args_pair["op_args"].items():
            print(f"        {key.ljust(20)}: {value}")

        print("  kernel_args:")
        for key, value in args_pair["kernel_args"].items():
            print(f"        {key.ljust(20)}: {value}")

        print()


if __name__ == "__main__":
    """
    usage: python3 ./experiments/op_param_kernel_block_analysis.py
    """

    node_kernel_pairs = tfp.get_pairs_from_trace_file("./examples/yolov8n-orto0.json")
    relation_2_instance_dict = mrf.get_relationship_dict(node_kernel_pairs)

    op_name = "Conv"
    provider = "CUDAExecutionProvider"
    kernel_sequence = (
        "void cudnn::winograd::generateWinogradTilesKernel<0, float, float>(cudnn::winograd::GenerateWinogradTilesParams<float, float>)",
        "_5x_cudnn_volta_scudnn_winograd_128x128_ldg1_ldg4_relu_tile148t_nt_v1",
        "void op_generic_tensor_kernel<3, float, float, float, 256, (cudnnGenericOp_t)0, (cudnnNanPropagation_t)0, 0>(cudnnTensorStruct, float*, cudnnTensorStruct, float const*, cudnnTensorStruct, float const*, float, float, float, cudnnActivationStruct, reducedDivisorArray, int)"
    )

    relation_tuple = (op_name, provider, kernel_sequence)
    kernel_idx = 1

    console_report(relation_2_instance_dict, relation_tuple, kernel_idx)