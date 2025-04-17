import onnx
import argparse
import os
from utils import trace_file_parser as tfp

def __set_kernel_attributes(kernel_node, kernel):
    """
    此函数的作用是为内核节点添加属性。

    Args:
        kernel_node (onnx.NodeProto): 要添加属性的 ONNX 节点。
        kernel (Json 对象): 包含内核属性的对象。

    Returns:
        onnx.NodeProto: 添加属性后的内核节点。
    """
    dur_attr = onnx.helper.make_attribute("duration", kernel["dur"])
    kernel_node.attribute.append(dur_attr)
    idx_attr = onnx.helper.make_attribute("index", kernel["Index"])
    kernel_node.attribute.append(idx_attr)
    block_x_attr = onnx.helper.make_attribute("block_x", kernel["args"]["block_x"])
    kernel_node.attribute.append(block_x_attr)
    block_y_attr = onnx.helper.make_attribute("block_y", kernel["args"]["block_y"])
    kernel_node.attribute.append(block_y_attr)
    block_z_attr = onnx.helper.make_attribute("block_z", kernel["args"]["block_z"])
    kernel_node.attribute.append(block_z_attr)
    grid_x_attr = onnx.helper.make_attribute("grid_x", kernel["args"]["grid_x"])
    kernel_node.attribute.append(grid_x_attr)
    grid_y_attr = onnx.helper.make_attribute("grid_y", kernel["args"]["grid_y"])
    kernel_node.attribute.append(grid_y_attr)
    grid_z_attr = onnx.helper.make_attribute("grid_z", kernel["args"]["grid_z"])
    kernel_node.attribute.append(grid_z_attr)
    return kernel_node

def __set_kernel_parent_attribute(kernel_node, parent_node, parent_trace_node):
    """
    此函数的作用是为内核节点添加父节点属性。

    Args:
        kernel_node (onnx.NodeProto): 要添加属性的 ONNX 节点。
        parent_node (onnx.NodeProto): 父节点。
        parent_trace_node (dict): 实测运行中包含父节点属性的对象。

    Returns:
        onnx.NodeProto: 添加属性后的内核节点。
    """

    # 从 ONNX 图中父节点获取参数
    new_attr = onnx.helper.make_attribute("parent_name", parent_node.name)
    kernel_node.attribute.append(new_attr)
    for attr in parent_node.attribute:
        attr.name = f"p_{attr.name}"
        kernel_node.attribute.append(attr)

    # 从实测运行中获取父节点属性
    new_attr = onnx.helper.make_attribute("p_trace_duration", parent_trace_node["dur"])
    kernel_node.attribute.append(new_attr)
    new_attr = onnx.helper.make_attribute("p_trace_output_size", int(parent_trace_node["args"]["output_size"]))
    kernel_node.attribute.append(new_attr)
    new_attr = onnx.helper.make_attribute("p_trace_parameter_size", int(parent_trace_node["args"]["parameter_size"]))
    kernel_node.attribute.append(new_attr)
    new_attr = onnx.helper.make_attribute("p_trace_activation_size", int(parent_trace_node["args"]["activation_size"]))
    kernel_node.attribute.append(new_attr)
    for idx, input in enumerate(parent_trace_node["args"]["input_type_shape"]):
        input_shape = list(input.values())[0]
        if input_shape:
            new_attr = onnx.helper.make_attribute(f"p_trace_input_shape_{idx}", input_shape)
        else:
            new_attr = onnx.helper.make_attribute(f"p_trace_input_shape_{idx}", [-1])
            print(f"[onnx_filler] {parent_node.name} 的输入 {idx} 形状为空")
        kernel_node.attribute.append(new_attr)
    
    return kernel_node

def replace_operators_with_kernels(onnx_model, node_kernel_mapping):
    """
    该函数的功能是将 ONNX 模型中的操作符替换为对应的内核节点。
    将算子级 ONNX 计算图转换为 kernel 级计算图，主要方便通过 Netron 观察与分析，无法运行。

    Args:
        onnx_model (onnx.ModelProto): 要处理的 ONNX 模型。
        node_kernel_mapping (dict): 节点名称到内核信息的映射字典。

    Returns:
        onnx.ModelProto: 替换操作符后的 ONNX 模型。
    """
    graph = onnx_model.graph
    nodes_to_remove = []
    new_nodes = []

    for node in graph.node:
        # 遍历所有 ONNX 图中节点
        search_name = f"{node.name}_kernel_time"
        if search_name in node_kernel_mapping:
            # 有执行记录的节点（算子）
            trace_pair = node_kernel_mapping[search_name]
            kernels = trace_pair["Kernels"]
            trace_node = trace_pair["Node"]
            if kernels:
                # 存在对应的 kernel 序列
                prev_outputs = None
                for idx, kernel in enumerate(kernels):
                    kernel_name = f"{node.name}/kernel_{idx}"

                    # 对于第一个 kernel，输入为原节点的输入；其余为前继 kernel 的输出
                    if idx == 0:
                        kernel_inputs = node.input
                    else:
                        kernel_inputs = prev_outputs

                    # 对于最后一个 kernel，输出为原节点的输出；其余为新建一个对应的输出
                    if idx == len(kernels) - 1:
                        kernel_outputs = node.output
                    else:
                        kernel_outputs = [f"{kernel_name}_output"]

                    kernel_node = onnx.helper.make_node(
                        kernel.get('name', 'Unknown'),
                        kernel_inputs,
                        kernel_outputs,
                        name=kernel_name
                    )
                    
                    kernel_node = __set_kernel_attributes(kernel_node, kernel)
                    kernel_node = __set_kernel_parent_attribute(kernel_node, node, trace_node)
                    new_nodes.append(kernel_node)

                    prev_outputs = kernel_outputs
                nodes_to_remove.append(node)
            else:
                print(f"[onnx_filler] {node.name} 算子被执行但无对应 kernel")
        else:
            print(f"[onnx_filler] {node.name} 未找到对应 _kernel_time，算子未发送至 CUDA EP")

    # 移除需要替换的节点
    for node in nodes_to_remove:
        graph.node.remove(node)

    # 添加新生成的节点
    graph.node.extend(new_nodes)

    return onnx_model

def add_kernels_for_operators(onnx_model, node_kernel_mapping):
    """
    此函数的作用是为 ONNX 模型中的操作符添加对应的内核节点，并为操作符节点添加索引和持续时间属性。
    但是使用 Netron 观察效果不好。

    Args:
        onnx_model (onnx.ModelProto): 待处理的 ONNX 模型。
        node_kernel_mapping (dict): 节点名称到内核信息的映射字典，键为节点名称加上 "_kernel_time"，值为包含内核信息的字典。

    Returns:
        onnx.ModelProto: 处理后的 ONNX 模型，其中添加了内核节点，并为操作符节点添加了属性。
    """
    graph = onnx_model.graph

    for node in graph.node:
        # 遍历图中所有算子
        search_name = f"{node.name}_kernel_time"
        if search_name in node_kernel_mapping:
            # 存在执行记录的节点（算子）
            trace_pair = node_kernel_mapping[search_name]
            kernels = trace_pair["Kernels"]
            trace_node = trace_pair["Node"]
            if kernels:
                # 由 CUDA EP 执行并且实际发射了 kernel
                kernel_inputs = []
                for idx, kernel in enumerate(kernels):
                    kernel_name = f"{node.name}/kernel_{idx}"
                    kernel_outputs = [f"{kernel_name}_output"]

                    kernel_node = onnx.helper.make_node(
                        kernel.get('name', 'Unknown'),
                        kernel_inputs,
                        kernel_outputs,
                        name=kernel_name
                    )

                    kernel_node = __set_kernel_attributes(kernel_node, kernel)
                    kernel_node = __set_kernel_parent_attribute(kernel_node, node, trace_node)
                    graph.node.append(kernel_node)

                    kernel_inputs = kernel_outputs
                
                node.input.append(kernel_outputs[0])
            else:
                print(f"[onnx_filler] {node.name} 算子被执行但无对应 kernel")
        else:
            print(f"[onnx_filler] {node.name} 未找到对应 _kernel_time，算子未发送至 CUDA EP")

        node.attribute.append(onnx.helper.make_attribute("index", trace_pair["Node"]["Index"]))
        node.attribute.append(onnx.helper.make_attribute("duration", trace_pair["Node"]["dur"]))
    
    return onnx_model


def main():
    parser = argparse.ArgumentParser(description='Fill ONNX model with kernel information.')
    parser.add_argument('onnx', type=str, help='Path to the input ONNX file')
    parser.add_argument('trace', type=str, help='Path to the trace file')
    parser.add_argument('--output', type=str, help='Path to the output ONNX file')
    parser.add_argument('--mode', type=str, help='Mode of filling, either "replace" or "add"', default="replace")

    args = parser.parse_args()

    if args.output is None:
        base_name, ext = os.path.splitext(args.onnx)
        args.output = f"{base_name}_kernel{ext}"

    onnx_model = onnx.load(args.onnx)
    node_kernel_pairs = tfp.get_pairs_from_trace_file(args.trace)
    node_kernel_mapping = tfp.get_node_kernel_mapping(node_kernel_pairs)

    if args.mode == "replace":
        filled_model = replace_operators_with_kernels(onnx_model, node_kernel_mapping)
    elif args.mode == "add":
        filled_model = add_kernels_for_operators(onnx_model, node_kernel_mapping)
    else:
        print(f"[onnx_filler] 未知模式: {args.mode}")
        exit(1)

    onnx.save(filled_model, args.output)
    print(f"[onnx_filler] 成功将填充后的模型保存到 {args.output}")


if __name__ == "__main__":
    """
    Usage:
        简单的运行：
            python3 ./onnx_kernel_filler.py ./examples/yolov8n.onnx ./examples/yolov8n-orto0.json
    """
    main()