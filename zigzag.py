def zigzag_traversal(matrix):
    rows, cols = len(matrix), len(matrix[0])
    zigzag = []  # 用于存储结果
    for sum_idx in range(rows + cols - 1):  # 按对角线的和来遍历
        if sum_idx % 2 == 0:  # 偶数和 -> 从上到下遍历
            row = min(sum_idx, rows - 1)  # 起始行
            col = sum_idx - row  # 起始列
            while row >= 0 and col < cols:
                zigzag.append(matrix[row][col])
                row -= 1
                col += 1
        else:  # 奇数和 -> 从下到上遍历
            col = min(sum_idx, cols - 1)  # 起始列
            row = sum_idx - col  # 起始行
            while col >= 0 and row < rows:
                zigzag.append(matrix[row][col])
                col -= 1
                row += 1
    two_d_array = [zigzag[i:i + cols] for i in range(0, len(zigzag), cols)]
    return two_d_array


# 使用示例
matrix = [
    [1, 2, 3, 4],
    [5, 6, 7, 8],
    [9, 10, 11, 12],
    [13, 14, 15, 16],
]

# 调用函数并打印结果
zigzag_result = zigzag_traversal(matrix)
print(zigzag_result)
