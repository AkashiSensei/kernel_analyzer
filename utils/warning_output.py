import sys
import warnings

YELLOW = '\033[93m'
RED = '\033[91m'
ENDC = '\033[0m'

def _simple_warning_impl(message, category, filename, lineno, file=None, line=None):
    if file is None:
        file = sys.stderr
    print(f"{YELLOW}{message}{ENDC}", file=file)

def _detailed_warning_impl(message, category, filename, lineno, file=None, line=None):
    if file is None:
        file = sys.stderr
    print(f"{YELLOW}{filename}: {lineno}: {category.__name__}", file=file)
    print(f"{message}{ENDC}", file=file)

def _detailed_error_impl(message, category, filename, lineno, file=None, line=None):
    if file is None:
        file = sys.stderr
    print(f"{RED}{filename}: {lineno}: {category.__name__}", file=file)
    print(f"{message}{ENDC}", file=file)

# 对外暴露的警告触发函数（只需一行调用）
def simple(message):
    original_showwarning = warnings.showwarning  # 保存原始输出函数
    warnings.showwarning = _simple_warning_impl  # 临时设置自定义输出
    warnings.warn(message, UserWarning)  # 触发警告
    warnings.showwarning = original_showwarning  # 恢复原始输出（避免影响后续）

def detailed(message):
    original_showwarning = warnings.showwarning
    warnings.showwarning = _detailed_warning_impl
    warnings.warn(message, UserWarning)
    warnings.showwarning = original_showwarning

def error(message, exit_code=1):
    original_showwarning = warnings.showwarning
    warnings.showwarning = _detailed_error_impl
    warnings.warn(message, UserWarning)
    warnings.showwarning = original_showwarning
    sys.exit(exit_code)