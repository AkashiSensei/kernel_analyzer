from rule_based_model import trace_file_based_tester as tfbt

def single_source_test():
    model_exact_table = {
        '规则库数据来源计算图': ['V8n', 'V8s', 'V8m', 'V8l', 'V8x']
    }
    model_seq_table = {
        '规则库数据来源计算图': ['V8n', 'V8s', 'V8m', 'V8l', 'V8x']
    }
    conv_exact_table = {
        '规则库数据来源计算图': ['V8n', 'V8s', 'V8m', 'V8l', 'V8x']
    }
    conv_seq_table = {
        '规则库数据来源计算图': ['V8n', 'V8s', 'V8m', 'V8l', 'V8x']
    }

    source_model_type_list = ["n", "s", "m", "l", "x"]
    target_model_type_list = ["n", "s", "m", "l", "x"]

    for target_model_type in target_model_type_list:
        model_exact_table[f'V8{target_model_type}'] = []
        model_seq_table[f'V8{target_model_type}'] = []
        conv_exact_table[f'V8{target_model_type}'] = []
        conv_seq_table[f'V8{target_model_type}'] = []

    for source_model_type in source_model_type_list:
        for target_model_type in target_model_type_list:
            predicting_trace_file_path = f"./results/trace/yolov8-orto0/yolov8{target_model_type}-orto0.json"
            rule_model_data_path = f"./rule_based_model/data/single-yolov8/yolov8{source_model_type}-orto0-ncu.json"

            model_exact_acc, model_seq_acc = tfbt.trace_file_based_test(predicting_trace_file_path, rule_model_data_path)
            conv_exact_acc, conv_seq_acc = tfbt.trace_file_based_test(predicting_trace_file_path, rule_model_data_path, ['Conv'])

            model_exact_table[f'V8{target_model_type}'].append(model_exact_acc)
            model_seq_table[f'V8{target_model_type}'].append(model_seq_acc)
            conv_exact_table[f'V8{target_model_type}'].append(conv_exact_acc)
            conv_seq_table[f'V8{target_model_type}'].append(conv_seq_acc)

    print("model_exact_table: ")
    print(model_exact_table)
    print("model_seq_table: ")
    print(model_seq_table)
    print("conv_exact_table: ")
    print(conv_exact_table)
    print("conv_seq_table: ")
    print(conv_seq_table)

def multi_source_test():
    source_model_type_list = ["n_s", "s_l", "l_x", "m_l_x"]
    target_model_type_list = ["n", "s", "m", "l", "x"]

    model_exact_table = {
        '规则库数据来源计算图': [
            f"v8{source_model_type}" for source_model_type in source_model_type_list
        ]
    }
    model_seq_table = {
        '规则库数据来源计算图': [
            f"v8{source_model_type}" for source_model_type in source_model_type_list
        ]
    }

    for target_model_type in target_model_type_list:
        model_exact_table[f'V8{target_model_type}'] = []
        model_seq_table[f'V8{target_model_type}'] = []

    for source_model_type in source_model_type_list:
        for target_model_type in target_model_type_list:
            predicting_trace_file_path = f"./results/trace/yolov8-orto0/yolov8{target_model_type}-orto0.json"
            rule_model_data_path = f"./rule_based_model/data/multi-yolov8/yolov8{source_model_type}-orto0-ncu.json"

            model_exact_acc, model_seq_acc = tfbt.trace_file_based_test(predicting_trace_file_path, rule_model_data_path)

            model_exact_table[f'V8{target_model_type}'].append(model_exact_acc)
            model_seq_table[f'V8{target_model_type}'].append(model_seq_acc)

    print("model_exact_table: ")
    print(model_exact_table)
    print("model_seq_table: ")
    print(model_seq_table)

if __name__ == '__main__':
    """
    Usage: python3 ./experiments/kernels_accuracy.py
    """
    # single_source_test()
    multi_source_test()