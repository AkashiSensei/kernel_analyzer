from typing import List
import numpy
from onnx_tool.tensor import Tensor
import onnx_tool
import os

def profile_model(model_path, input_shape: tuple[int], save=False):
    """
    使用 onnx-tool 对模型进行静态分析，输出分析报告，可选将其保存。

    Args:
        `model_path` (str): 模型路径。
        `input_shape` (tuple[int]): 输入形状。
        `save` (bool, optional): 是否保存分析报告。默认为 False。
    """
    m = onnx_tool.Model(model_path)
    m.graph.shape_infer({'data': numpy.zeros(input_shape)})  # update tensor shapes with new input tensor
    m.graph.profile()
    m.graph.print_node_map()  # console print

    if not save:
        return

    # 获取输入文件名，建立输出保存目录
    model_name = os.path.splitext(os.path.basename(model_path))[0]
    model_dir = os.path.dirname(model_path)
    sub_dir_path = os.path.join(model_dir, model_name + "_analyse")
    if not os.path.exists(sub_dir_path):
        os.mkdir(sub_dir_path)
    text_path = os.path.join(sub_dir_path, "report.txt")
    csv_path = os.path.join(sub_dir_path, "report.csv")
    analysed_model_path = os.path.join(sub_dir_path, model_name + "_analysed.onnx")

    # 保存
    m.graph.print_node_map(text_path)  # save file
    m.graph.print_node_map(csv_path)
    m.save_model(analysed_model_path)


if __name__ == '__main__':
    """
    usage: python3 ./rule_based_model/data_shape_analyzer.py
    """
    # example yolov8n
    model_path = "./examples/yolov8n.onnx"
    input_shape = (1, 3, 640, 640)

    profile_model(model_path, input_shape, save=True)
