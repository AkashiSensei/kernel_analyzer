import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from matplotlib.colors import Normalize, FuncNorm

compute_throughput_table = {'规则库数据来源计算图': ['V8n', 'V8m', 'V8x', 'V8n_m', 'V8n_x', 'V8m_x'], 'V8n': [0.9681733841463884, 0.828178219595675, 0.7928179759014654, 0.9681733841463884, 0.9681733841463884, 0.8436830406411858], 'V8s': [0.8281313081888996, 0.8991718168028031, 0.8376203773090164, 0.9029313184877481, 0.8784255041529313, 0.8989377137558872], 'V8m': [0.7895133746425761, 0.9739859345615663, 0.8713113335035522, 0.9747401221094997, 0.8812315938122616, 0.9739859345615663], 'V8l': [0.7519574320666277, 0.9181160310903183, 0.9021438138380546, 0.923384577469204, 0.9186328881666889, 0.9230339389538856], 'V8x': [0.7572902631876328, 0.9093888957568954, 0.9726485875733736, 0.914371166615094, 0.9724125879767808, 0.9729528705720939]}
memory_throughput_table = {'规则库数据来源计算图': ['V8n', 'V8m', 'V8x', 'V8n_m', 'V8n_x', 'V8m_x'], 'V8n': [0.968134227271473, 0.8103964025245114, 0.762326343178666, 0.968134227271473, 0.968134227271473, 0.8226709873890597], 'V8s': [0.7917313784414389, 0.8759053499681259, 0.8160689277494722, 0.8845445666410242, 0.8597389632645968, 0.8829068908844002], 'V8m': [0.7255466324351765, 0.9731710031637346, 0.8262141169980391, 0.9732857870854434, 0.8462311648606373, 0.9731710031637346], 'V8l': [0.6637386322720058, 0.8850798413489619, 0.8794114941118875, 0.8877309720766264, 0.8960078968199321, 0.898906279894955], 'V8x': [0.6731182344649842, 0.8732284355937849, 0.9759847606158892, 0.8799306108590428, 0.975497336179494, 0.9752124088763183]}
sm_active_cycles_table = {'规则库数据来源计算图': ['V8n', 'V8m', 'V8x', 'V8n_m', 'V8n_x', 'V8m_x'], 'V8n': [0.9688737883795471, 0.8001299202434341, 0.7954514934164906, 0.9688737883795471, 0.9688737883795471, 0.8136377442898592], 'V8s': [0.7873052937412153, 0.8335150935633101, 0.767596256499898, 0.868876637342271, 0.8411697347111599, 0.8304233434291735], 'V8m': [0.6826786960451364, 0.9737863047691939, 0.790253824092941, 0.9730781727134481, 0.819497649094804, 0.9737863047691939], 'V8l': [0.5986264290911258, 0.8443687984804321, 0.8207205169168942, 0.8474416972055994, 0.8285319571425143, 0.8383760557694179], 'V8x': [0.5778214609722223, 0.7978767089115071, 0.9797180077323667, 0.8024612578294625, 0.9788469850644167, 0.9792039787709209]}

def draw(data, output_path, start_value):
# 创建 DataFrame
    df = pd.DataFrame(data)
    df.set_index('规则库数据来源计算图', inplace=True)

    # 将分数转换为百分比
    def convert_to_percentage(value):
        return value * 100
    for col in df.columns:
        df[col] = df[col].apply(convert_to_percentage)

    # 设置字体
    plt.rcParams['font.family'] = 'SimSun'
    # 增大字体大小
    plt.rcParams['font.size'] = 36

    # 定义 100% 时的颜色
    max_color = '#3a090b'

    # 创建颜色映射
    cmap = sns.light_palette(max_color, as_cmap=True)

    # 指定最浅色对应的值
    min_value = start_value
    max_value = 100

    # 创建自定义的归一化对象
    norm = Normalize(vmin=min_value, vmax=max_value)

    # 调整图形尺寸，可根据实际情况调整
    plt.figure(figsize=(10, 10))

    # 绘制热图，同时调整注释字体大小和加粗，精细调整块之间的间距
    # 调整颜色条与图的间距
    sns.heatmap(df, annot=True, fmt=".1f", cmap=cmap, linewidths=5, annot_kws={"size": 36, "weight": "bold"},
                cbar_kws={"pad": 0.03}, norm=norm)

    # 添加横轴标题，并设置加粗，调整横轴标题和标签的距离
    plt.xlabel('被预测目标计算图', fontweight='bold', labelpad=10)
    # 设置纵轴标签并加粗，调整纵轴标题和标签的距离
    plt.ylabel('规则库数据来源计算图', fontweight='bold', labelpad=10)

    # 调整横轴刻度标签与图的间距
    plt.tick_params(axis='x', pad=10)
    # 调整纵轴刻度标签与图的间距
    plt.tick_params(axis='y', pad=10)

    # 将y轴刻度标签设置为水平显示
    plt.yticks(rotation=45,  ha='right', fontsize=28)

    # 调整子图布局，为标题和坐标轴留出更多空间
    plt.subplots_adjust(left=0.17, bottom=0.14, right=1, top=0.95)

    # 移除大标题
    # 保存图片，设置 dpi 提高清晰度
    plt.savefig(output_path, dpi=100)
    plt.close()

if __name__ == '__main__':
    """
    Usage: python3 ./plotter/yolov8_mix_data_kernels_execute_accuracy.py
    """
    draw(compute_throughput_table, './plotter/output/yolov8_mix_compute_throughput.png', 75)
    draw(memory_throughput_table, './plotter/output/yolov8_mix_memory_throughput.png', 65)
    draw(sm_active_cycles_table, './plotter/output/yolov8_mix_sm_active_cycles.png', 55)
