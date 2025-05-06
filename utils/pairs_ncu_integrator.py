from utils import warning_output as wout
from utils import trace_file_parser as tfp
import pandas as pd

def _fill_kernel_metric(kernel, id_df, metric_name):
    filtered_df = id_df[id_df['Metric Name'] == metric_name]

    # 检查结果数量
    if len(filtered_df) == 0:
        wout.simple(f"[pairs_ncu_integrator] Kernel ID: {kernel['Index']} 中未找到 {metric_name} 的数据。")
        return 
    elif len(filtered_df) > 1:
        wout.simple(f"[pairs_ncu_integrator] Kernel ID: {kernel['Index']} 中找到多条 {metric_name} 的数据。")
        return

    # 获取 "Metric Value" 和 "Metric Unit" 两列的值
    metric_value = filtered_df['Metric Value'].values[0]
    metric_unit = filtered_df['Metric Unit'].values[0]

    kernel["ncu"][metric_name + " Value"] = metric_value
    kernel["ncu"][metric_name + " Unit"] = metric_unit



def fill_pairs_with_ncu(node_kernel_pairs, ncu_csv_path):
    try:
        df = pd.read_csv(ncu_csv_path)
    except FileNotFoundError:
        wout.error("[pairs_ncu_integrator]NCU CSV file not found.")
    
    kernel_idx = 0
    
    for pair in node_kernel_pairs:
        for kernel in pair["Kernels"]:
            kernel_idx = kernel["Index"]
            # 跳过不计数的 kernel
            if kernel_idx < 0:
                continue


            # 从 csv 中找到对应 ID 的数据并校验 name
            id_df = df[df["ID"] == kernel_idx]
            if id_df.empty:
                wout.error(f"[pairs_ncu_integrator] Kernel ID {kernel_idx} not found in NCU CSV.")
            
            # 将数据填入 pair
            kernel["ncu"] = {}
            
            # GPU Speed Of Light Throughput
            _fill_kernel_metric(kernel, id_df, "Compute (SM) Throughput")
            _fill_kernel_metric(kernel, id_df, "Memory Throughput")
            _fill_kernel_metric(kernel, id_df, "SM Active Cycles")
            _fill_kernel_metric(kernel, id_df, "Elapsed Cycles")

            # Launch Statistics
            _fill_kernel_metric(kernel, id_df, "Registers Per Thread")
            _fill_kernel_metric(kernel, id_df, "# SMs")
            _fill_kernel_metric(kernel, id_df, "Shared Memory Configuration Size")  # "Shared Memory executed" in Nsys
            _fill_kernel_metric(kernel, id_df, "Driver Shared Memory Per Block")
            _fill_kernel_metric(kernel, id_df, "Dynamic Shared Memory Per Block")
            _fill_kernel_metric(kernel, id_df, "Static Shared Memory Per Block")

            # Occupancy
            _fill_kernel_metric(kernel, id_df, "Block Limit SM")
            _fill_kernel_metric(kernel, id_df, "Block Limit Registers")
            _fill_kernel_metric(kernel, id_df, "Block Limit Shared Mem")
            _fill_kernel_metric(kernel, id_df, "Block Limit Warps")
            _fill_kernel_metric(kernel, id_df, "Theoretical Active Warps per SM")
            _fill_kernel_metric(kernel, id_df, "Theoretical Occupancy")
            _fill_kernel_metric(kernel, id_df, "Achieved Occupancy")
            _fill_kernel_metric(kernel, id_df, "Achieved Active Warps Per SM")

    print(f"[pairs_ncu_integrator] Processed kernel count: {kernel_idx + 1}")
    if kernel_idx != df["ID"].max():
        wout.error(f"[pairs_ncu_integrator] Kernel count not match. Ncu: {df['ID'].max() + 1}, Tracing: {kernel_idx + 1}.")

    return node_kernel_pairs

if __name__ == "__main__":
    """
    Test:
        python3 ./utils/pairs_ncu_integrator.py
    """

    file_path = "./examples/yolov8n-orto0.json"
    ncu_csv_path = "./examples/ncu/yolov8n-orto0-ncu-basic.csv"

    node_kernel_pairs = tfp.get_pairs_from_trace_file(file_path)
    node_kernel_pairs = fill_pairs_with_ncu(node_kernel_pairs, ncu_csv_path)

    print(node_kernel_pairs[0]["Kernels"][0]["ncu"])