from rule_based_model import kernel_execute_metric_tester as kemt

def mix_source_test():
    source_model_type_list = ["n", "m", "x", "n_m", "n_x", "m_x"]
    target_model_type_list = ["n", "s", "m", "l", "x"]

    compute_throughput_table = {
        '规则库数据来源计算图': [
            f"V8{source_type}" for source_type in source_model_type_list
        ]
    }
    memory_throughput_table = {
        '规则库数据来源计算图': [
            f"V8{source_type}" for source_type in source_model_type_list
        ]
    }
    sm_active_cycles_table = {
        '规则库数据来源计算图': [
            f"V8{source_type}" for source_type in source_model_type_list
        ]
    }

    for target_model_type in target_model_type_list:
        compute_throughput_table[f'V8{target_model_type}'] = []
        memory_throughput_table[f'V8{target_model_type}'] = []
        sm_active_cycles_table[f'V8{target_model_type}'] = []

    for source_model_type in source_model_type_list:
        for target_model_type in target_model_type_list:
            predicting_trace_file_path = f"./results/trace/yolov8-orto0/yolov8{target_model_type}-orto0.json"
            predicting_ncu_csv_path = f"./results/ncu/retest-yolov8/yolov8{target_model_type}-orto0-ncu-basic.csv"
            if len(source_model_type) == 1:
                rule_model_data_path = f"./rule_based_model/data/single-yolov8/yolov8{source_model_type}-orto0-ncu.json"
            else:
                rule_model_data_path = f"./rule_based_model/data/multi-yolov8/yolov8{source_model_type}-orto0-ncu.json"

            compute_acc, memory_acc, sm_cycle_acc = kemt.GPU_performance_teste(predicting_trace_file_path, predicting_ncu_csv_path, rule_model_data_path)

            compute_throughput_table[f'V8{target_model_type}'].append(compute_acc)
            memory_throughput_table[f'V8{target_model_type}'].append(memory_acc)
            sm_active_cycles_table[f'V8{target_model_type}'].append(sm_cycle_acc)

    print("compute_throughput_table: ")
    print(compute_throughput_table)
    print("memory_throughput_table: ")
    print(memory_throughput_table)
    print("sm_active_cycles_table: ")
    print(sm_active_cycles_table)
    
    return compute_throughput_table, memory_throughput_table, sm_active_cycles_table

if __name__ == '__main__':
    """
    Usage: python3 ./experiments/kernel_execute_accuracy.py
    """
    mix_source_test()