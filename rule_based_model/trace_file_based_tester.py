from rule_based_model import data_based_kernel_finder as dbkf
from utils import warning_output as wout
from utils import trace_file_parser as tfp

def judge_kernel_match_verbose(result_kernels, real_kernels):
    """
    判断 kernel 序列和真实值是否一致，并给出详细输出。

    Args:
        `result_kernels` (list): 作为预测结果的 kernel 序列列表，None 标识未能给出预测。
        `real_kernels` (list): 真实 kernel 序列列表。
    
    Returns:
        "no_match": 不匹配
        "exact": 精确匹配
        "sequence": 序列一致，但具体线程块等的大小存在差异
    """
    sequence_match = True
    exact_match = True

    # 检查非空
    if result_kernels is None:
        wout.simple("[trace_file_based_tester] 没有给出预测结果")
        return "no_match"

    # 判断长度
    if len(result_kernels) != len(real_kernels):
        wout.simple(f"[trace_file_based_tester] 预测结果与真实结果长度不一致，预测序列长度为 {len(result_kernels)}，真实结果长度为 {len(real_kernels)}")
        return "no_match"

    # 判断每一个 kernel
    for result_kernel, real_kernel in zip(result_kernels, real_kernels):
        if result_kernel["name"] != real_kernel["name"]:
            wout.simple("[trace_file_based_tester] 预测结果与真实结果 kernel 不一致")
            sequence_match = False
    
    if not sequence_match:
        print(f"预测序列：")
        for result_kernel in result_kernels:
            print("  " + result_kernel["name"])
        print(f"真实序列：")
        for real_kernel in real_kernels:
            print("  " + real_kernel["name"])
        return "no_match"
    
    print(f"[trace_file_based_tester] 预测结果与真实结果序列一致，对比每一个 kernel：")
    for result_kernel, real_kernel in zip(result_kernels, real_kernels):
        print(f"  {result_kernel['name']}")
        if "Memcpy" in result_kernel["name"]:
            continue
        
        # 对比线程块和网格形状
        result_grid = [result_kernel["args"]["grid_x"], result_kernel["args"]["grid_y"], result_kernel["args"]["grid_z"]]
        real_grid = [real_kernel["args"]["grid_x"], real_kernel["args"]["grid_y"], real_kernel["args"]["grid_z"]]
        result_block = [result_kernel["args"]["block_x"], result_kernel["args"]["block_y"], result_kernel["args"]["block_z"]]
        real_block = [real_kernel["args"]["block_x"], real_kernel["args"]["block_y"], real_kernel["args"]["block_z"]]
        
        if result_grid == real_grid:
            print(f"    grid    :{str(result_grid).ljust(25)}:{str(real_grid).ljust(25)}")
        else:
            exact_match = False
            wout.simple(f"    grid    :{str(result_grid).ljust(25)}:{str(real_grid).ljust(25)}")
    
        if result_block == real_block:
            print(f"    block   :{str(result_block).ljust(25)}:{str(real_block).ljust(25)}")
        else:
            exact_match = False
            wout.simple(f"    block   :{str(result_block).ljust(25)}:{str(real_block).ljust(25)}")
    
    if exact_match:
        print("[trace_file_based_tester] kernel 精确匹配")
        return "exact"
    else:
        wout.simple("[trace_file_based_tester] kernel 序列匹配")
        return "sequence"

    

def trace_file_based_test(predicting_trace_file_path, rule_model_data_path, op_list=None):
    """
    使用跟踪文件来测试基于规则的分析模型，并给出详细输出。

    Args:
        `predicting_trace_file_path` (str): 预测的跟踪文件路径。
        `rule_model_data_path` (str): 规则模型数据路径。
        `op_list` (list): 运算符列表，默认为 None，表示使用所有运算符（除内存操作）。
    """

    predicting_node_kernel_pairs = tfp.get_pairs_from_trace_file(predicting_trace_file_path)
    rule_model_data = dbkf.load_json_data(rule_model_data_path)

    # 准确率计数器
    total_node_cnt = 0
    sequence_match_node_cnt = 0
    exact_match_node_cnt = 0
    
    for pair in predicting_node_kernel_pairs:
        node = pair["Node"]
        real_kernels = pair["Kernels"]

        op_name = node['args']['op_name']
        node_name = node['name']

        if "Memcpy" in op_name:
            continue

        if op_list is not None and op_name not in op_list:
            continue

        print(f"[trace_file_based_tester] 找到 {op_name} 算子：{node_name} ，准备开始预测，算子信息如下")
        total_node_cnt += 1

        # 构建用于预测的数据 node_info
        predicting_node_info = {
            "op_name": op_name,
            "node_name": node_name,
        }
        for idx, input_type_shape in enumerate(node["args"]["input_type_shape"]):
            input_shape = list(input_type_shape.values())[0]
            predicting_node_info[f"input_shape_{idx}"] = input_shape
        for idx, output_type_shape in enumerate(node["args"]["output_type_shape"]):
            output_shape = list(output_type_shape.values())[0]
            predicting_node_info[f"output_shape_{idx}"] = output_shape

        # 输出 node 数据
        for key, value in predicting_node_info.items():
            print(f"{key.ljust(25)}: {value}")
        
        result_kernels = dbkf.find_best_match_kernels(rule_model_data, predicting_node_info)
        
        # 检查并输出结果
        match = judge_kernel_match_verbose(result_kernels, real_kernels)
        if match == "exact":
            exact_match_node_cnt += 1
            sequence_match_node_cnt += 1
        elif match == "sequence":
            sequence_match_node_cnt += 1
        
        print()
        print()
        
    print(f"精确匹配: {exact_match_node_cnt} / {total_node_cnt}")
    print(f"序列匹配: {sequence_match_node_cnt} / {total_node_cnt}")

if __name__ == "__main__":
    """
    usage: python3 ./rule_based_model/trace_file_based_tester.py
    """
    predicting_trace_file_path = "./examples/yolov8n-orto0.json"  # 作为预测对象，从中提取算子进行预测，以及正确结果作为验证
    rule_model_data_path = "./rule_based_model/data/yolov8n-orto0-origin.json"  # 规则模型的数据
    op_list = ["Reshape"]

    trace_file_based_test(predicting_trace_file_path, rule_model_data_path)