from PIL import Image
import cv2
import numpy as np
import random
import pandas as pd

# 打开灰度图片
image = Image.open('pic.jpg').convert('L')
image_matrix = np.array(image)
print(image_matrix)


# 加载图像
def load_image(image_path):
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)  # 灰度图像
    if img is None:
        raise ValueError("无法加载图像")
    return img

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

    return zigzag

# 将图像分割成小块（子图像）
# 将图像分割成小块（子图像）
def split_image(img, block_size):
    h, w = img.shape  # 获取图像的高度和宽度
    blocks = []

    # 如果高度或宽度不是 block_size 的整数倍，则进行填充
    if h % block_size != 0 or w % block_size != 0:
        new_h = (h // block_size + 1) * block_size
        new_w = (w // block_size + 1) * block_size

        # 创建一个填充后的图像
        padded_img = np.zeros((new_h, new_w), dtype=img.dtype)
        padded_img[:h, :w] = img
    else:
        padded_img = img.copy()

    # 对填充后的图像按块分割
    for i in range(0, padded_img.shape[0], block_size):
        for j in range(0, padded_img.shape[1], block_size):
            block = padded_img[i:i + block_size, j:j + block_size]
            blocks.append(block)

    return blocks

def zigzag_blocks(blocks):
    zigzag_results = []
    for block in blocks:
        zigzag_results.append(zigzag_traversal(block))
    return zigzag_results



# 对每个子图像进行加密（这里采用随机像素置乱的方式）
#def encrypt_block(block):
#    block_flat = block.flatten()
 #   random.shuffle(block_flat)  # 对像素值进行随机置乱
  #  return block_flat.reshape(block.shape)

def encrypt_block(block, data_array):
    block_flat = block.flatten()

    # 使用混沌序列对像素值进行置乱
    encrypted_block_flat = block_flat.copy()
    encrypt = data_array[:len(block_flat)]
    for i in range(len(block_flat)):
        encrypted_block_flat[i] = block_flat[np.where(encrypt == i)[0][0]]

    return encrypted_block_flat.reshape(block.shape)


# 重新拼接加密后的子图像
def combine_blocks(blocks, img_shape, block_size):
    h, w = img_shape
    encrypted_img = np.zeros((h, w), dtype=np.uint8)
    idx = 0
    for i in range(0, h, block_size):
        for j in range(0, w, block_size):
            encrypted_img[i:i + block_size, j:j + block_size] = blocks[idx]
            idx += 1
    return encrypted_img


# 展示图像
def display_image(img, title="Image"):
    cv2.imshow(title, img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


# 主函数
def main():
    # 加载原图像
    image_path = 'pic.jpg'  # 替换为你的图像路径
    img = load_image(image_path)

    # 子图像大小
    block_size = 32  # 可以根据需要调整子图像大小

    # 读取混沌序列
    df = pd.read_csv('chen_max.csv', header=None)
    data_array = df.to_numpy()

    # 分割图像为小块
    blocks = split_image(img, block_size)

    # 对每个子图像进行加密
    encrypted_blocks = [encrypt_block(block, data_array) for block in blocks]  # 这里传递data_array

    # 重新拼接加密后的子图像
    encrypted_img = combine_blocks(encrypted_blocks, img.shape, block_size)

    # 显示原图和加密后的图像
    display_image(img, "Original Image")
    display_image(encrypted_img, "Encrypted Image")



if __name__ == '__main__':
    main()
