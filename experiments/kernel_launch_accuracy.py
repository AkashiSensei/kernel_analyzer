from rule_based_model import GPU_perfermance_tester as gpt

def single_source_test():
    grid_size_table = {
        '规则库数据来源模型': ['V8n', 'V8s', 'V8m', 'V8l', 'V8x']
    }
    block_size_table = {
        '规则库数据来源模型': ['V8n', 'V8s', 'V8m', 'V8l', 'V8x']
    }
    register_size_table = {
        '规则库数据来源模型': ['V8n', 'V8s', 'V8m', 'V8l', 'V8x']
    }

    source_model_type_list = ["n", "s", "m", "l", "x"]
    target_model_type_list = ["n", "s", "m", "l", "x"]

    for target_model_type in target_model_type_list:
        grid_size_table[f'V8{target_model_type}'] = []
        block_size_table[f'V8{target_model_type}'] = []
        register_size_table[f'V8{target_model_type}'] = []

    for source_model_type in source_model_type_list:
        for target_model_type in target_model_type_list:
            predicting_trace_file_path = f"./results/trace/yolov8-orto0/yolov8{target_model_type}-orto0.json"
            predicting_ncu_csv_path = f"./results/ncu/ultralytics-yolov8/yolov8{target_model_type}-orto0-ncu-basic.csv"
            rule_model_data_path = f"./rule_based_model/data/single-yolov8/yolov8{source_model_type}-orto0-ncu.json"

            gs_acc, bs_acc, r_acc = gpt.GPU_performance_teste(predicting_trace_file_path, predicting_ncu_csv_path, rule_model_data_path)

            grid_size_table[f'V8{target_model_type}'].append(gs_acc)
            block_size_table[f'V8{target_model_type}'].append(bs_acc)
            register_size_table[f'V8{target_model_type}'].append(r_acc)

    print(grid_size_table)
    print(block_size_table)
    print(register_size_table)
    
    return grid_size_table, block_size_table, register_size_table

if __name__ == '__main__':
    """
    Usage: python3 ./experiments/kernel_launch_accuracy.py
    """
    single_source_test()