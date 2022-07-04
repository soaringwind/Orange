import numpy as np
import matplotlib.pyplot as plt

img = plt.imread("./cv/coins.jpg")
img = np.array(img)
print(img.shape)

# Dimension Reduction
a, b, c = img.shape
new_img = np.zeros(shape=(a, b))
print(new_img.shape)
for i in range(a):
    for j in range(b):
        new_img[i, j] = img[i, j].mean()

# 2. Edge Dectection: y: [[-1, -1, -1], [0, 0, 0], [1, 1, 1]], x: [[-1, 0, 1], [-1, 0, 1], [-1, 0, 1]]
# [
# [new_img[i-1, j-1], new_img[i-1, j], new_img[i-1, j+1]], 
# [new_img[i, j-1], new_img[i, j], new_img[i, j+1]], 
# [new_img[i+1, j-1], new_img[i+1, j], new_img[i+1, j+1]]
# ]
conv_img = np.zeros(shape=(a, b))
conv_theta = np.zeros(shape=(a, b))
conv_x = np.array([[1, 0, -1], [1, 0, -1], [1, 0, -1]])
conv_y = np.array([[1, 1, 1], [0, 0, 0], [-1, -1, -1]])
for i in range(1, a-1):
    for j in range(1, b-1):
        cell = np.array([
        [new_img[i-1, j-1], new_img[i-1, j], new_img[i-1, j+1]], 
        [new_img[i, j-1], new_img[i, j], new_img[i, j+1]], 
        [new_img[i+1, j-1], new_img[i+1, j], new_img[i+1, j+1]]
        ])
        conv_img_x = np.sum(conv_x * cell)
        conv_img_y = np.sum(conv_y * cell) # 卷积使用对应元素点乘
        conv_img[i, j] = np.sqrt(conv_img_x**2+conv_img_y**2)
        conv_theta[i, j] = np.arctan(conv_img_y / conv_img_x)
high_threshold = 50
low_threshold = 20
high_img = conv_img.copy()
low_img = conv_img.copy()
# 抑制孤立边缘
# 1. 直接检查每个点周围是否有超过高门限的点，有留下，没有置0
# for i in range(1, a-1):
#     for j in range(1, b-1):
#         cell = np.array([
#         [low_img[i-1, j-1], low_img[i-1, j], low_img[i-1, j+1]], 
#         [low_img[i, j-1], low_img[i, j], low_img[i, j+1]], 
#         [low_img[i+1, j-1], low_img[i+1, j], low_img[i+1, j+1]]
#         ])
#         low_img[i, j] = low_img[i, j] if low_img[i, j] > low_threshold else 0
#         low_img[i, j] = np.max(cell) if np.max(cell) >= high_threshold and low_img[i, j] != 0 else 0
# 2. 使用图的算法来进行检查
visited = [] # 用来计算所有遍历过的点
stack = [] # 用来记录中心点
queue = [] # 用来记录弱边缘
for i in range(1, a-1):
    for j in range(1, b-1):
        if (i, j) in visited:
            continue
        visited.append((i, j))
        if high_img[i ,j] < low_threshold:
            high_img[i, j] = 0
            continue
        if high_img[i, j] >= high_threshold:
            high_img[i , j] = high_threshold
            continue
        stack.append((i, j))
        queue.append((i, j))
        connected = False
        while stack:
            t_i, t_j = stack.pop(0)
            if high_img[t_i, t_j] >= high_threshold:
                connected = True

            # 检查八个邻域是否有弱边缘
            if high_img[t_i-1, t_j-1] > low_threshold and (t_i-1, t_j-1) not in queue:
                stack.append((t_i-1, t_j-1))
                queue.append((t_i-1, t_j-1))
            if high_img[t_i-1, t_j] > low_threshold and (t_i-1, t_j) not in queue:
                stack.append((t_i-1, t_j))
                queue.append((t_i-1, t_j))
            if high_img[t_i-1, t_j+1] > low_threshold and (t_i-1, t_j+1) not in queue:
                stack.append((t_i-1, t_j+1))
                queue.append((t_i-1, t_j+1))
            if high_img[t_i, t_j-1] > low_threshold and (t_i, t_j-1) not in queue:
                stack.append((t_i, t_j-1))
                queue.append((t_i, t_j-1))
            if high_img[t_i, t_j+1] > low_threshold and (t_i, t_j+1) not in queue:
                stack.append((t_i, t_j+1))
                queue.append((t_i, t_j+1))
            if high_img[t_i+1, t_j-1] > low_threshold and (t_i+1, t_j-1) not in queue:
                stack.append((t_i+1, t_j-1))
                queue.append((t_i+1, t_j-1))
            if high_img[t_i+1, t_j] > low_threshold and (t_i+1, t_j) not in queue:
                stack.append((t_i+1, t_j))
                queue.append((t_i+1, t_j))
            if high_img[t_i+1, t_j+1] > low_threshold and (t_i+1, t_j+1) not in queue:
                stack.append((t_i+1, t_j+1))
                queue.append((t_i+1, t_j+1))
        while queue:
            i_, j_ = queue.pop(-1)
            visited.append((i_, j_))
            high_img[i_, j_] = high_threshold if connected else 0