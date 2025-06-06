import argparse
from collections import defaultdict
from utils import trace_file_parser as tfp


def main():
    parser = argparse.ArgumentParser(description='Analyze JSON file for operator - kernel mappings.')
    parser.add_argument('--input', required=True, help='Path to input JSON file')
    parser.add_argument('--names', action='store_true',
                        help='Show node names for each occurrence')
    parser.add_argument('--imem', action='store_true',
                        help='Ignore kernels containing memcpy (case - insensitive)')
    parser.add_argument('--md', action='store_true',
                        help='Output analysis results in Markdown format。')
    args = parser.parse_args()

    result = defaultdict(lambda: {'count': 0, 'nodes': []})
    # 用于记录每个 (op_name, provider) 出现的总次数
    op_provider_count = defaultdict(int)
    # 用于记录每个 (op_name, provider) 有非空 kernel 序列对应的次数
    op_provider_non_empty_kernel_count = defaultdict(int)

    node_kernel_pairs = tfp.get_pairs_from_trace_file(args.input)
    if node_kernel_pairs is None:
        return

    for pair in node_kernel_pairs:
        current_node = pair["Node"]
        current_kernels = pair["Kernels"]

        op_name = current_node['args']['op_name']
        provider = current_node['args']['provider']
        node_name = current_node['name']
        kernel_sequence = [kernel['name'] for kernel in current_kernels]

        if args.imem:
            kernel_sequence = [k for k in kernel_sequence if 'memcpy' not in k.lower()]

        key = (op_name, provider, tuple(kernel_sequence))
        result[key]['count'] += 1
        result[key]['nodes'].append(node_name)
        op_provider_count[(op_name, provider)] += 1
        if kernel_sequence:
            op_provider_non_empty_kernel_count[(op_name, provider)] += 1

    # 按 (op_name, provider) 分组
    grouped_result = defaultdict(list)
    for (op_name, provider, kernel_tuple), info in result.items():
        grouped_result[(op_name, provider)].append((kernel_tuple, info))

    # 按出现次数降序排序
    for op_provider in grouped_result:
        grouped_result[op_provider].sort(key=lambda x: x[1]['count'], reverse=True)

    # 统计每个 (op_name, provider) 对应的 kernel 序列种类数
    op_provider_kernel_sequence_count = {
        op_provider: len(mappings)
        for op_provider, mappings in grouped_result.items()
    }

    # 简化 provider 名称
    def simplify_provider(provider):
        return provider.replace('ExecutionProvider', '')

    # 输出统计信息
    cuda_ops = []
    cpu_ops = []
    for (op_name, provider) in sorted(op_provider_count.keys()):
        if provider == 'CUDAExecutionProvider':
            cuda_ops.append((op_name, provider))
        elif provider == 'CPUExecutionProvider':
            cpu_ops.append((op_name, provider))

    if args.md:
        if cuda_ops:
            print("## 各算子出现次数统计")
            print("### CUDA 执行提供程序")
            print("| 算子 - 执行提供程序 | 总出现次数 | 有非空 Kernel 序列对应次数 | 对应的 Kernel 序列种类数 |")
            print("| ---- | ---- | ---- | ---- |")
            for (op_name, provider) in cuda_ops:
                count = op_provider_count[(op_name, provider)]
                non_empty = op_provider_non_empty_kernel_count[(op_name, provider)]
                seq_count = op_provider_kernel_sequence_count.get((op_name, provider), 0)
                simplified_provider = simplify_provider(provider)
                print(f"| {op_name} - {simplified_provider} | {count} | {non_empty} | {seq_count} |")
            print()

        if cpu_ops:
            print("### CPU 执行提供程序")
            print("| 算子 - 执行提供程序 | 总出现次数 |")
            print("| ---- | ---- |")
            for (op_name, provider) in cpu_ops:
                count = op_provider_count[(op_name, provider)]
                simplified_provider = simplify_provider(provider)
                print(f"| {op_name} - {simplified_provider} | {count} |")
            print()
    else:
        if cuda_ops:
            print("各算子（CUDA 执行提供程序）出现次数统计：")
            for (op_name, provider) in cuda_ops:
                count = op_provider_count[(op_name, provider)]
                non_empty = op_provider_non_empty_kernel_count[(op_name, provider)]
                seq_count = op_provider_kernel_sequence_count.get((op_name, provider), 0)
                simplified_provider = simplify_provider(provider)
                print(f"Operator: {op_name} - {simplified_provider}, 总出现次数: {count}, 有非空 Kernel 序列对应次数: {non_empty}, 对应的 Kernel 序列种类数: {seq_count}")
            print()

        if cpu_ops:
            print("各算子（CPU 执行提供程序）出现次数统计：")
            for (op_name, provider) in cpu_ops:
                count = op_provider_count[(op_name, provider)]
                simplified_provider = simplify_provider(provider)
                print(f"Operator: {op_name} - {simplified_provider}, 总出现次数: {count}")
            print()

    # 按 Kernel 序列种数降序排列
    sorted_op_providers = sorted(grouped_result.keys(), key=lambda x: op_provider_kernel_sequence_count[x], reverse=True)

    # 输出结果
    first_op = True
    if args.md:
        for op_provider in sorted_op_providers:
            op_name, provider = op_provider
            if provider == 'CPUExecutionProvider':
                continue
            simplified_provider = simplify_provider(provider)
            print(f"## {op_name} - {simplified_provider} ({op_provider_non_empty_kernel_count[op_provider]}/{op_provider_count[op_provider]})")
            mappings = grouped_result[op_provider]
            for index, (kernel_tuple, info) in enumerate(mappings, start=1):
                if len(kernel_tuple) == 0:
                    print(f"### 无对应 Kernel（{info['count']} 次）")
                else:
                    print(f"### 第 {index} 组对应关系（{info['count']}次）")
                    print(f"Kernel sequence (len: {len(kernel_tuple)}):")
                    for kernel in kernel_tuple:
                        print(f"`{kernel}`")
                if args.names:
                    print("Node names:")
                    for node in info['nodes']:
                        print(f"`{node}`")
    else:
        for op_provider in sorted_op_providers:
            op_name, provider = op_provider
            if provider == 'CPUExecutionProvider':
                continue
            simplified_provider = simplify_provider(provider)
            if not first_op:
                print("\n\n")
            first_op = False
            print(f"Operator: {op_name} - {simplified_provider}")
            print(f"  Count: {op_provider_count[op_provider]}")
            print(f"  Non empty: {op_provider_non_empty_kernel_count[op_provider]}")
            mappings = grouped_result[op_provider]
            for index, (kernel_tuple, info) in enumerate(mappings, start=1):
                if index > 1:
                    print()
                if len(kernel_tuple) == 0:
                    print(f"  无对应 Kernel")
                    print(f"    Count: {info['count']}")
                else:
                    print(f"  第 {index} 组对应关系:")
                    print(f"    Count: {info['count']}")
                    print(f"    Kernel sequence length: {len(kernel_tuple)}")
                    print("    Kernel sequence:")
                    for kernel in kernel_tuple:
                        print(f"      {kernel}")
                if args.names:
                    print("    Node names:")
                    for node in info['nodes']:
                        print(f"      {node}")


if __name__ == "__main__":
    """
    usage: python3 ./experiments/trace_kernel_reporter.py --input ./examples/yolov8n-orto0.json --md
    """
    main()
    