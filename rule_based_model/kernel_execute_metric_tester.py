import sys
from utils import trace_file_parser as tfp
from utils import pairs_ncu_integrator as pni
from utils import warning_output as wout
from rule_based_model import GPU_performance_calculator as gpc
from rule_based_model import data_based_kernel_finder as dbkf

# 0 - 不输出
# 1 - 输出报错
# 2 - 输出警告
# 3 - 输出基本信息
# 4 - 输出详细信息

output_level = 2

def cal_data_acc(predict_result, real_value):
    """
    计算准确率，一方占另一方的百分比，取不大于 1 的值

    args:
        `predict_result`: 数值形式的预测值
        `real_value`: 数值形式的真实值

    return:
        double: 预测值和真实值之间的准确率
    """
    if predict_result == real_value:
        return 1.0
    if predict_result == 0 or real_value == 0:
        return 0.0
    return min(predict_result / real_value, real_value / predict_result)

def output(output_text, level):
    if level <= output_level:
        print(output_text)

def warning(output_text):
    if 2 <= output_level:
        wout.simple(output_text)

def error(output_text, exit_code=1):
    if 1 <= output_level:
        wout.error(output_text, exit_code=exit_code)
    else:
        sys.exit(exit_code)
        
        
def GPU_performance_teste(predicting_trace_file_path, predicting_ncu_csv_path, rule_model_data_path):

    predicting_node_kernel_pairs = tfp.get_pairs_from_trace_file(predicting_trace_file_path)
    predicting_node_kernel_pairs = pni.fill_pairs_with_ncu(predicting_node_kernel_pairs, predicting_ncu_csv_path)
    rule_model_data = dbkf.load_json_data(rule_model_data_path)

    # 输出相关信息
    output(f"[kernel_execute_metric_tester] 待测跟踪文件路径：{predicting_trace_file_path}", 3)
    output(f"[kernel_execute_metric_tester] 待测 NCU 文件路径：{predicting_ncu_csv_path}", 3)
    output(f"[kernel_execute_metric_tester] 规则模型数据文件路径：{rule_model_data_path}", 3)



    # 准确率相关计数器
    # 通用
    operators_count = 0
    # kernel 启动信息
    compute_throughput_acc_sum = 0.0
    memory_throughput_acc_sum = 0.0
    sm_active_cycles_acc_sum = 0.0


    # 以算子为单位进行指标计算
    for pair in predicting_node_kernel_pairs:
        node = pair["Node"]
        real_kernels = pair["Kernels"]
        op_name = node['args']['op_name']
        node_name = node['name']
        # 跳过内存操作节点，虽然不应该有
        if "Memcpy" in op_name:
            continue


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
    
        output(f"[kernel_execute_metric_tester] 找到 {op_name} 算子：{node_name} ，开始预测", 3)
        for key, value in predicting_node_info.items():
            output(f"{key.ljust(25)}: {value}", 4)

        
        # 预测
        result_kernels = dbkf.find_best_match_kernels(rule_model_data, predicting_node_info)
        if result_kernels == None:
            warning(f"[kernel_execute_metric_tester] 没有找到算子 {op_name} 对应的 kernel 序列")


        # 计算准确率
        operators_count += 1
        compute_throughput_acc = cal_data_acc(
            gpc.get_op_metric_average(result_kernels, gpc.get_kernel_compute_throughput),
            gpc.get_op_metric_average(real_kernels, gpc.get_kernel_compute_throughput),
        )
        compute_throughput_acc_sum += compute_throughput_acc
        memory_throughput_acc = cal_data_acc(
            gpc.get_op_metric_average(result_kernels, gpc.get_kernel_memory_throughput),
            gpc.get_op_metric_average(real_kernels, gpc.get_kernel_memory_throughput),
        )
        memory_throughput_acc_sum += memory_throughput_acc
        sm_active_cycles_acc = cal_data_acc(
            gpc.get_op_metric_sum(result_kernels, gpc.get_kernel_sm_active_cycles),
            gpc.get_op_metric_sum(real_kernels, gpc.get_kernel_sm_active_cycles),
        )
        sm_active_cycles_acc_sum += sm_active_cycles_acc
        output("该算子参数准确率如下：", 4)
        output(f"compute_throughput_acc: {compute_throughput_acc}", 4)
        output(f"memory_throughput_acc: {memory_throughput_acc}", 4)
        output(f"sm_active_cycles_acc: {sm_active_cycles_acc}", 4)

    # 输出整体准确率
    output("整体参数准确率如下：", 0)
    output(f"compute_throughput_acc: {compute_throughput_acc_sum / operators_count}", 0)
    output(f"memory_throughput_acc: {memory_throughput_acc_sum / operators_count}", 0)
    output(f"sm_active_cycles_acc: {sm_active_cycles_acc_sum / operators_count}", 0)

    return compute_throughput_acc_sum / operators_count, memory_throughput_acc_sum / operators_count, sm_active_cycles_acc_sum / operators_count


if __name__ == "__main__":
    """
    Usage: python3 ./rule_based_model/kernel_execute_metric_tester.py
    """
    # 测试实验
    predicting_trace_file_path="./results/trace/yolov8-orto0/yolov8l-orto0.json"
    predicting_ncu_csv_path="./results/ncu/ultralytics-yolov8/yolov8l-orto0-ncu-basic.csv"
    rule_model_data_path="./rule_based_model/data/single-yolov8/yolov8s-orto0-ncu.json"

    GPU_performance_teste(predicting_trace_file_path, predicting_ncu_csv_path, rule_model_data_path)