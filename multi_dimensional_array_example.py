import numpy as np

# 創建一個 3x3 的多維陣列
array_1 = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
print("Array 1:")
print(array_1)

# 創建另一個 3x3 的多維陣列
array_2 = np.array([[9, 8, 7], [6, 5, 4], [3, 2, 1]])
print("\nArray 2:")
print(array_2)

# 多維陣列的加法運算
sum_array = array_1 + array_2
print("\nSum of Array 1 and Array 2:")
print(sum_array)

# 多維陣列的減法運算
diff_array = array_1 - array_2
print("\nDifference of Array 1 and Array 2:")
print(diff_array)

# 多維陣列的乘法運算（逐元素相乘）
prod_array = array_1 * array_2
print("\nProduct of Array 1 and Array 2 (element-wise):")
print(prod_array)

# 多維陣列的矩陣乘法
matmul_array = np.dot(array_1, array_2)
print("\nMatrix multiplication of Array 1 and Array 2:")
print(matmul_array)

# 多維陣列的轉置
transpose_array = array_1.T
print("\nTranspose of Array 1:")
print(transpose_array)
