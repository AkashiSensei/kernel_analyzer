from utils import trace_file_parser as tfp
from utils import mapping_relation_info_finder as mrif

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

    instance_pairs = mrif.get_pairs_from_single_relation_dict(relation_2_instance_dict, relation_tuple)
    args_pairs = mrif.get_args_from_pairs(instance_pairs, kernel_idx)

    print(f"Trace file: {trace_file_path}")
    print(f"Op: {relation_tuple[0]}")
    print(f"EP: {relation_tuple[1]}")
    print(f"Kernels: {relation_tuple[2]}")
    print(f"Kernel idx: {kernel_idx}")
    print(f"Kernel name: {relation_tuple[2][kernel_idx]}")
    print()
    print()

    # 按照 `output_size` 排序
    args_pairs.sort(key=lambda x: x["op_args"]["output_size"])

    # for idx, args_pair in enumerate(args_pairs):
    #     print(f"Instance {idx + 1}/{len(args_pairs)}")
    #     print("  op_args:")
    #     for key, value in args_pair["op_args"].items():
    #         print(f"        {key.ljust(20)}: {value}")

    #     print("  kernel_args:")
    #     for key, value in args_pair["kernel_args"].items():
    #         print(f"        {key.ljust(20)}: {value}")

    #     print()

    for idx, args_pair in enumerate(args_pairs):
        print(f"Instance {idx + 1}/{len(args_pairs)}")
        op_args = args_pair["op_args"]
        kernel_args = args_pair["kernel_args"]

        # 找出 op_args 和 kernel_args 中最长的键长度
        max_op_key_length = max(len(key) for key in op_args.keys()) + 2 if op_args else 0
        max_kernel_key_length = max(len(key) for key in kernel_args.keys()) + 2 if kernel_args else 0

        # 找出 op_args 和 kernel_args 中最长的值长度
        max_op_value_length = max(len(str(value)) for value in op_args.values()) + 2 if op_args else 0
        max_kernel_value_length = max(len(str(value)) for value in kernel_args.values()) + 2 if kernel_args else 0

        # 转换为列表以便遍历
        op_items = list(op_args.items())
        kernel_items = list(kernel_args.items())

        # 确定最大长度
        max_length = max(len(op_items), len(kernel_items))

        # 计算左右两侧内容的总宽度
        total_width = (max_op_key_length + max_op_value_length + 2) + (max_kernel_key_length + max_kernel_value_length + 2) + 3

        # 定义标题
        op_title = "op_args"
        kernel_title = "kernel_args"

        # 计算标题需要的空格数以实现居中对齐
        op_title_width = max_op_key_length + max_op_value_length + 2
        kernel_title_width = max_kernel_key_length + max_kernel_value_length + 2
        op_title_padding = (op_title_width - len(op_title)) // 2
        kernel_title_padding = (kernel_title_width - len(kernel_title)) // 2

        # 打印居中对齐的标题
        print(f"  {' ' * op_title_padding}{op_title}{' ' * (op_title_width - len(op_title) - op_title_padding)}|  {' ' * kernel_title_padding}{kernel_title}{' ' * (kernel_title_width - len(kernel_title) - kernel_title_padding)}")

        for i in range(max_length):
            op_item = op_items[i] if i < len(op_items) else (None, None)
            kernel_item = kernel_items[i] if i < len(kernel_items) else (None, None)

            if op_item[0] is not None:
                op_value_str = str(op_item[1])
                op_str = f"{op_item[0].ljust(max_op_key_length)}: {op_value_str.ljust(max_op_value_length)}"
            else:
                op_str = " " * (max_op_key_length + max_op_value_length + 2)

            if kernel_item[0] is not None:
                kernel_value_str = str(kernel_item[1])
                kernel_str = f"{kernel_item[0].ljust(max_kernel_key_length)}: {kernel_value_str.ljust(max_kernel_value_length)}"
            else:
                kernel_str = ""

            # 输出左右并列的信息
            print(f"  {op_str}|  {kernel_str}")

        print()


if __name__ == "__main__":
    """
    usage: python3 ./experiments/op_param_kernel_block_analysis.py
    """

    # # Conv - winograd
    # trace_file_path = "./examples/yolov8n-orto0.json"
    # op_name = "Conv"
    # provider = "CUDAExecutionProvider"
    # kernel_sequence = (
    #     "void cudnn::winograd::generateWinogradTilesKernel<0, float, float>(cudnn::winograd::GenerateWinogradTilesParams<float, float>)",
    #     "_5x_cudnn_volta_scudnn_winograd_128x128_ldg1_ldg4_relu_tile148t_nt_v1",
    #     "void op_generic_tensor_kernel<3, float, float, float, 256, (cudnnGenericOp_t)0, (cudnnNanPropagation_t)0, 0>(cudnnTensorStruct, float*, cudnnTensorStruct, float const*, cudnnTensorStruct, float const*, float, float, float, cudnnActivationStruct, reducedDivisorArray, int)",
    # )
    # kernel_idx = 1


    # Concat
    trace_file_path = "./examples/yolov8n-orto0.json"
    op_name = "Concat"
    provider = "CUDAExecutionProvider"
    # Concat same
    # kernel_sequence = (
    #     "void onnxruntime::cuda::_ConcatKernelSameConcatDim<int, onnxruntime::cuda::TArray<void const*, 32> >(onnxruntime::cuda::DivMod<int>, onnxruntime::cuda::DivMod<int>, onnxruntime::cuda::DivMod<int>, int*, onnxruntime::cuda::TArray<void const*, 32>, int)",
    # )
    # kernel_idx = 0
    # Concat normal
    kernel_sequence = (
        "MemcpyHostToDevice",
        "MemcpyHostToDevice",
        "MemcpyHostToDevice",
        "MemcpyHostToDevice",
        "void onnxruntime::cuda::_ConcatKernel<int>(onnxruntime::cuda::DivMod<int>, onnxruntime::cuda::DivMod<int>, long const*, long const*, long const*, int*, void const**, int)"
    )
    kernel_idx = 4


    node_kernel_pairs = tfp.get_pairs_from_trace_file(trace_file_path)
    relation_2_instance_dict = mrif.get_relationship_dict(node_kernel_pairs)

    relation_tuple = (op_name, provider, kernel_sequence)
    

    console_report(relation_2_instance_dict, relation_tuple, kernel_idx)