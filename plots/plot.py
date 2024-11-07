import numpy as np
import matplotlib.pyplot as plt
import sys
import os

# 獲取當前腳本的絕對路徑，並將上一層資料夾添加到 sys.path
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.abspath(os.path.join(current_dir, os.pardir))
sys.path.append(parent_dir)

from logic_functions import gradient_function  # 匯入 gradient_function

# 定義一個二次函數
def function_2(x):
    return x[0]**2 + x[1]**2

# 計算梯度
x = np.array([5.0, 3.6])
grad = gradient_function(function_2, x)

# 檢查梯度
print("Gradient at (5.0, 3.6):", grad)
