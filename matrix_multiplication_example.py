import numpy as np

# 創建兩個 2x3 和 3x2 的矩陣
matrix_1 = np.array([[1, 2, 3], [4, 5, 6]])
matrix_2 = np.array([[7, 8], [9, 10], [11, 12]])

print("Matrix 1:")
print(matrix_1)

print("\nMatrix 2:")
print(matrix_2)

# 矩陣乘積
product_matrix = np.dot(matrix_1, matrix_2)
print("\nProduct of Matrix 1 and Matrix 2:")
print(product_matrix)
