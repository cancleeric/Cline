
import numpy as np  # 添加这行代码来导入 numpy

def smooth_curve(x):
    """用於損失值的平滑曲線"""
    window_len = 11
    s = np.r_[x[window_len-1:0:-1], x, x[-2:-window_len-1:-1]]
    w = np.kaiser(window_len, 2)
    y = np.convolve(w/w.sum(), s, mode='valid')
    return y[5:len(y)-5]