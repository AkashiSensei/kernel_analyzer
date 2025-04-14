import os
from ultralytics import YOLO


def convert_pt_to_onnx(root_dir):
    for root, dirs, files in os.walk(root_dir):
        for file in files:
            if file.endswith('.pt'):
                pt_file_path = os.path.join(root, file)
                onnx_file_path = os.path.splitext(pt_file_path)[0] + '.onnx'
                try:
                    model = YOLO(pt_file_path)
                    success = model.export(format="onnx")
                    if success:
                        print(f"成功将 {pt_file_path} 转换为 {onnx_file_path}")
                    else:
                        print(f"转换 {pt_file_path} 失败")
                except Exception as e:
                    print(f"转换 {pt_file_path} 时出现错误: {e}")


if __name__ == "__main__":
    # 指定要遍历的目录
    target_directory = '.'
    convert_pt_to_onnx(target_directory)
    