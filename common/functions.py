import numpy as np

# 激活函數及其導數
def step(x):
    # 步階函數：輸入大於 0 時返回 1，否則返回 0
    return (x > 0).astype(int)

def sigmoid(x):
    # Sigmoid 函數：將輸入壓縮到 (0, 1) 範圍內
    return 1 / (1 + np.exp(-x))

def sigmoid_derivative(x):
    # Sigmoid 函數的導數
    return sigmoid(x) * (1 - sigmoid(x))

def relu(x):
    # ReLU 函數：輸入大於 0 時返回輸入值，否則返回 0
    return np.maximum(0, x)

def tanh(x):
    # Tanh 函數：雙曲正切函數
    return np.tanh(x)

def tanh_derivative(x):
    # Tanh 函數的導數
    return 1 - np.tanh(x) ** 2

def identity(x):
    # 恒等函數：返回輸入值本身
    return x

def softmax(x):
    """
    Softmax 函數：將輸入轉換為概率分佈
    """
    if x.ndim == 2:
        x = x - np.max(x, axis=1, keepdims=True)
        exp_x = np.exp(x)
        return exp_x / np.sum(exp_x, axis=1, keepdims=True)
    x = x - np.max(x)  # 減去最大值以避免溢出
    exp_x = np.exp(x)
    return exp_x / np.sum(exp_x)

# 損失函數
def mean_squared_error(y_true, y_pred):
    """
    計算均方誤差 (Mean Squared Error, MSE)
    
    參數:
    y_true -- 真實標籤
    y_pred -- 預測標籤
    
    返回:
    mse -- 均方誤差
    """
    mse = np.mean((y_true - y_pred) ** 2)
    return mse

# def cross_entropy_error(y_true, y_pred):
#     """
#     計算交叉熵誤差 (Cross Entropy Error)
    
#     參數:
#     y_true -- 真實標籤 (one-hot 編碼)
#     y_pred -- 預測標籤 (softmax 輸出)
    
#     返回:
#     cee -- 交叉熵誤差
#     """
#     # 小值避免 log(0)
#     epsilon = 1e-12
#     y_pred = np.clip(y_pred, epsilon, 1. - epsilon)

#     # 若 y_true 是 one-hot 編碼，轉換為正確標籤索引
#     if y_true.ndim == 1:
#         correct_class_prob = y_pred[np.arange(len(y_pred)), y_true]
#     else:
#         correct_class_prob = np.sum(y_true * np.log(y_pred), axis=1)

#     # 計算交叉熵誤差
#     cee = -np.mean(correct_class_prob)
#     return cee

def cross_entropy_error(y_true, y_pred):
    """
    計算交叉熵誤差 (Cross Entropy Error)
    
    參數:
    y_true -- 真實標籤 (one-hot 編碼)
    y_pred -- 預測標籤 (softmax 輸出)
    
    返回:
    cee -- 交叉熵誤差
    """
    epsilon = 1e-7
    y_pred = np.clip(y_pred, epsilon, 1. - epsilon)

    if y_true.ndim == 1:  # 當 y_true 是 one-hot 的索引編碼形式
        return -np.mean(np.log(y_pred[np.arange(len(y_pred)), y_true]))
    else:  # 當 y_true 是 one-hot 編碼形式
        return -np.mean(np.sum(y_true * np.log(y_pred), axis=1))

# 數值微分和梯度計算
def numerical_derivative(f, x, h=1e-5):
    """
    計算函數 f 在 x 點的數值微分
    :param f: 函數
    :param x: 自變量
    :param h: 微小變量
    :return: f 在 x 點的數值微分
    """
    return (f(x + h) - f(x - h)) / (2 * h)

def gradient_function(f, x, h=1e-5):
    """
    計算函數 f 在 x 點的梯度
    :param f: 函數
    :param x: 自變量 (numpy array)
    :param h: 微小變量
    :return: f 在 x 點的梯度 (numpy array)
    """
    x = np.array(x, dtype=float)  # 確保 x 是 float 類型，以避免類型問題
    grad = np.zeros_like(x)
    perturb = np.eye(len(x)) * h  # 單位矩陣乘以 h，用來生成擾動矩陣
    
    for i in range(len(x)):
        grad[i] = (f(x + perturb[i]) - f(x - perturb[i])) / (2 * h)
    
    return grad

def numerical_gradient(f, x, h=1e-5):
    """
    計算函數 f 在點 x 的數值梯度。

    參數:
    f -- 需要計算梯度的函數，輸入為 x，輸出為標量。
    x -- 自變量，作為計算梯度的點 (numpy array)。

    返回:
    grad -- 與 x 形狀相同的梯度 (numpy array)，包含 f 在每個元素上的偏導數。
    """
    # 0.0001，用來近似計算導數的小變量
    grad = np.zeros_like(x)  # 初始化梯度矩陣，與 x 形狀相同

    # 使用 np.nditer 遍歷 x 的每個元素
    it = np.nditer(x, flags=['multi_index'], op_flags=['readwrite'])
    while not it.finished:
        idx = it.multi_index  # 獲取當前元素的索引
        tmp_val = x[idx]  # 暫存當前索引的值
        
        # 計算 f(x + h)
        x[idx] = float(tmp_val) + h
        fxh1 = f(x)  # 當前索引增加 h 後的函數值
        
        # 計算 f(x - h)
        x[idx] = tmp_val - h
        fxh2 = f(x)  # 當前索引減少 h 後的函數值
        
        # 使用中心差分公式計算數值梯度
        grad[idx] = (fxh1 - fxh2) / (2 * h)
        
        # 還原 x 中當前索引的原始值
        x[idx] = tmp_val
        it.iternext()  # 移動到下一個索引
    
    return grad  # 返回計算出的梯度

# 梯度下降法
def gradient_descent(f, init_x, learning_rate=0.1, num_iterations=100):
    """
    使用梯度下降法最小化函數 f
    :param f: 目標函數
    :param init_x: 初始點 (numpy array)
    :param learning_rate: 學習率 (步長)
    :param num_iterations: 迭代次數
    :return: 優化後的 x 值和每次迭代的歷史 x 值
    """
    x = np.array(init_x, dtype=float)
    x_history = [x.copy()]  # 保存每次迭代的 x 值
    
    for _ in range(num_iterations):
        grad = gradient_function(f, x)
        x -= learning_rate * grad  # 更新 x
        x_history.append(x.copy())  # 保存每次迭代的 x 值
    
    return x, x_history

# 其他函數
def weighted_sum(x, weights, bias):
    # 加權和函數：計算輸入與權重的點積並加上偏置
    return np.dot(x, weights) + bias
