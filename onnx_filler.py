import json
import onnx
import argparse
import os
import trace_file_parser

def __set_kernel_attributes(kernel_node, kernel):
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

def replace_operators_with_kernels(onnx_model, node_kernel_mapping):
    graph = onnx_model.graph
    nodes_to_remove = []
    new_nodes = []

    for node in graph.node:
        search_name = f"{node.name}_kernel_time"
        if search_name in node_kernel_mapping:
            kernel_info = node_kernel_mapping[search_name]
            kernels = kernel_info["Kernels"]
            if kernels:
                prev_outputs = None
                for idx, kernel in enumerate(kernels):
                    kernel_name = f"{node.name}/kernel_{idx}"
                    if idx == 0:
                        kernel_inputs = node.input
                    else:
                        kernel_inputs = prev_outputs

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
                    
                    parent_attr = onnx.helper.make_attribute("parent", node.name)
                    kernel_node.attribute.append(parent_attr)
                    kernel_node = __set_kernel_attributes(kernel_node, kernel)
                    new_nodes.append(kernel_node)

                    prev_outputs = kernel_outputs
                nodes_to_remove.append(node)
            else:
                new_nodes.append(node)
                print(f"[onnx_filler] {node.name} 无对应 kernel")
        else:
            # 没有对应的 kernel 序列，保留原节点
            new_nodes.append(node)
            print(f"[onnx_filler] {node.name} 未找到对应 _kernel_time")

    # 移除需要替换的节点
    for node in nodes_to_remove:
        graph.node.remove(node)

    # 添加新生成的节点
    graph.node.extend(new_nodes)

    return onnx_model

def add_kernels_for_operators(onnx_model, node_kernel_mapping):
    graph = onnx_model.graph

    for node in graph.node:
        search_name = f"{node.name}_kernel_time"
        if search_name in node_kernel_mapping:
            kernel_info = node_kernel_mapping[search_name]
            kernels = kernel_info["Kernels"]
            if kernels:
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

                    parent_attr = onnx.helper.make_attribute("parent", node.name)
                    kernel_node.attribute.append(parent_attr)
                    kernel_node = __set_kernel_attributes(kernel_node, kernel)
                    graph.node.append(kernel_node)

                    kernel_inputs = kernel_outputs
                
                node.input.append(kernel_outputs[0])
            else:
                print(f"[onnx_filler] {node.name} 无对应 kernel")
        else:
            print(f"[onnx_filler] {node.name} 未找到对应 _kernel_time")

        node.attribute.append(onnx.helper.make_attribute("index", kernel_info["Node"]["Index"]))
        node.attribute.append(onnx.helper.make_attribute("duration", kernel_info["Node"]["dur"]))
    
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
    node_kernel_pairs = trace_file_parser.get_pairs_from_trace_file(args.trace)
    node_kernel_mapping = trace_file_parser.get_node_kernel_mapping(node_kernel_pairs)

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
    main()