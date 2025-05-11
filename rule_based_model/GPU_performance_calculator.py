from utils import warning_output as wout

def get_kernel_grid_size(kernel):
    if "Memcpy" in kernel["name"]:
        return 0

    grid_x = int(kernel["args"]["grid_x"])
    grid_y = int(kernel["args"]["grid_y"])
    grid_z = int(kernel["args"]["grid_z"])
    return grid_x * grid_y * grid_z

def get_kernel_block_size(kernel):
    if "Memcpy" in kernel["name"]:
        return 0
    
    block_x = int(kernel["args"]["block_x"])
    block_y = int(kernel["args"]["block_y"])
    block_z = int(kernel["args"]["block_z"])
    return block_x * block_y * block_z

def get_kernel_register_per_thread(kernel):
    if "Memcpy" in kernel["name"]:
        return 0
    
    return int(kernel["ncu"]["Registers Per Thread Value"])

def get_op_metric_sum(kernels, cal_func): 
    if kernels is None or len(kernels) == 0:
        return 0
    return sum(cal_func(kernel) for kernel in kernels)

def get_kernel_compute_throughput(kernel):
    if "Memcpy" in kernel["name"]:
        return 0
    
    return float(kernel["ncu"]["Compute (SM) Throughput Value"])

def get_kernel_memory_throughput(kernel):
    if "Memcpy" in kernel["name"]:
        return 0
    
    return float(kernel["ncu"]["Memory Throughput Value"])

def get_kernel_sm_active_cycles(kernel):
    if "Memcpy" in kernel["name"]:
        return 0
    
    return float(kernel["ncu"]["SM Active Cycles Value"])

def get_op_metric_average(kernels, cal_func): 
    if kernels is None or len(kernels) == 0:
        return 0
    return sum(cal_func(kernel) for kernel in kernels) / len(kernels)