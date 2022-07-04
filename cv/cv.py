from matplotlib.image import imsave
import numpy as np
import matplotlib.pyplot as plt


class CannyEdge(object):
    def __init__(self, img) -> None:
        self.img = img
        self.x, self.y = self.img.shape
        self.edge_img = np.zeros(shape=(self.x, self.y))
        self.sobel_x = np.array([[1, 0, -1], [2, 0, -2], [1, 0, -1]])
        self.sobel_y = np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]])
        self.theta = np.zeros(shape=(self.x, self.y))
        self.min_threshold = 40
        self.max_threshold = 80
        self.gaussian_template = np.array(
            [[0.0947416, 0.118318, 0.0947416], 
            [0.118318, 0.147761, 0.118318], 
            [0.0947416, 0.118318, 0.0947416]]
            )


    def gaussian_filter(self, gaussian_template=None):
        if not gaussian_template:
            gaussian_template = self.gaussian_template
        gaussian_img = conv_img(self.img, gaussian_template)
        gaussian_img[0, :] = gaussian_img[1, :]
        gaussian_img[:, 0] = gaussian_img[:, 1]
        gaussian_img[self.x-1, :] = gaussian_img[self.x-2, :]
        gaussian_img[:, self.y-1] = gaussian_img[:, self.y-2]
        plt.imsave('./gauss.jpg', gaussian_img, cmap="gray")
        return gaussian_img


    def edge_dect(self, gaussian_img):
        self.conv_img_x = conv_img(gaussian_img, self.sobel_x)
        self.conv_img_y = conv_img(gaussian_img, self.sobel_y)
        self.edge_img = np.sqrt(self.conv_img_x ** 2 + self.conv_img_y ** 2)
        self.theta = self.conv_img_y / self.conv_img_x
        self.nms()
        self.shappen_edge()
        return self.edge_img
        

    def nms(self):
        a, b = self.edge_img.shape
        for i in range(1, a-1):
            for j in range(1, b-1):
                if 0 <= self.theta[i, j] < np.pi/4 or -np.pi < self.theta[i, j] < -np.pi * 3 / 4:
                    up = (1-np.tan(abs(self.theta[i, j]))) * self.edge_img[i, j+1] + np.tan(abs(self.theta[i, j])) * self.edge_img[i-1, j+1]
                    down = (1-np.tan(abs(self.theta[i, j]))) * self.edge_img[i, j-1] + np.tan(abs(self.theta[i, j])) * self.edge_img[i+1, j-1]
                    self.edge_img[i, j] = self.edge_img[i, j] if self.edge_img[i, j] >= max(up, down) else 0
                if np.pi/4 <= self.theta[i, j] < np.pi/2 or -np.pi * 3 / 4 < self.theta[i, j] < -np.pi / 2:
                    up = (1-np.tan(1/abs(self.theta[i, j]))) * self.edge_img[i-1, j] + np.tan(1/abs(self.theta[i, j])) * self.edge_img[i-1, j+1]
                    down = (1-np.tan(1/abs(self.theta[i, j]))) * self.edge_img[i-1, j] + np.tan(1/abs(self.theta[i, j])) * self.edge_img[i+1, j-1]
                    self.edge_img[i, j] = self.edge_img[i, j] if self.edge_img[i, j] >= max(up, down) else 0
                if np.pi/2 <= self.theta[i, j] < np.pi*3/4 or -np.pi/2 < self.theta[i, j] < -np.pi/4:
                    up = (1-np.tan(1/abs(self.theta[i, j]))) * self.edge_img[i-1, j] + np.tan(1/abs(self.theta[i, j])) * self.edge_img[i-1, j-1]
                    down = (1-np.tan(1/abs(self.theta[i, j]))) * self.edge_img[i+1, j] + np.tan(1/abs(self.theta[i, j])) * self.edge_img[i+1, j+1]
                    self.edge_img[i, j] = self.edge_img[i, j] if self.edge_img[i, j] >= max(up, down) else 0
                if np.pi*3/4 <= self.theta[i, j] < np.pi or -np.pi/4 < self.theta[i, j] < 0:
                    up = (1-np.tan(abs(self.theta[i, j]))) * self.edge_img[i, j-1] + np.tan(abs(self.theta[i, j])) * self.edge_img[i-1, j-1]
                    down = (1-np.tan(abs(self.theta[i, j]))) * self.edge_img[i, j+1] + np.tan(abs(self.theta[i, j])) * self.edge_img[i+1, j+1]
                    self.edge_img[i, j] = self.edge_img[i, j] if self.edge_img[i, j] >= max(up, down) else 0
        return 


    def shappen_edge(self):
        a, b = self.edge_img.shape
        visited = set() # 用来计算所有遍历过的点，使用集合大大提高效率
        stack = [] # 用来记录中心点
        queue = [] # 用来记录弱边缘
        for i in range(1, a-1):
            for j in range(1, b-1):
                if (i, j) in visited:
                    continue
                visited.add((i, j))
                if self.edge_img[i ,j] < self.min_threshold:
                    self.edge_img[i, j] = 0
                    continue
                if self.edge_img[i, j] >= self.max_threshold:
                    self.edge_img[i , j] = self.max_threshold
                    continue
                stack.append((i, j))
                queue.append((i, j))
                connected = False
                while stack:
                    t_i, t_j = stack.pop(0)
                    if self.edge_img[t_i, t_j] >= self.max_threshold:
                        connected = True

                    # 检查八个邻域是否有弱边缘
                    if self.edge_img[t_i-1, t_j-1] > self.min_threshold and (t_i-1, t_j-1) not in queue:
                        stack.append((t_i-1, t_j-1))
                        queue.append((t_i-1, t_j-1))
                    if self.edge_img[t_i-1, t_j] > self.min_threshold and (t_i-1, t_j) not in queue:
                        stack.append((t_i-1, t_j))
                        queue.append((t_i-1, t_j))
                    if self.edge_img[t_i-1, t_j+1] > self.min_threshold and (t_i-1, t_j+1) not in queue:
                        stack.append((t_i-1, t_j+1))
                        queue.append((t_i-1, t_j+1))
                    if self.edge_img[t_i, t_j-1] > self.min_threshold and (t_i, t_j-1) not in queue:
                        stack.append((t_i, t_j-1))
                        queue.append((t_i, t_j-1))
                    if self.edge_img[t_i, t_j+1] > self.min_threshold and (t_i, t_j+1) not in queue:
                        stack.append((t_i, t_j+1))
                        queue.append((t_i, t_j+1))
                    if self.edge_img[t_i+1, t_j-1] > self.min_threshold and (t_i+1, t_j-1) not in queue:
                        stack.append((t_i+1, t_j-1))
                        queue.append((t_i+1, t_j-1))
                    if self.edge_img[t_i+1, t_j] > self.min_threshold and (t_i+1, t_j) not in queue:
                        stack.append((t_i+1, t_j))
                        queue.append((t_i+1, t_j))
                    if self.edge_img[t_i+1, t_j+1] > self.min_threshold and (t_i+1, t_j+1) not in queue:
                        stack.append((t_i+1, t_j+1))
                        queue.append((t_i+1, t_j+1))
                while queue:
                    i_, j_ = queue.pop(-1)
                    visited.add((i_, j_))
                    self.edge_img[i_, j_] = self.max_threshold if connected else 0
        plt.imsave("./edge.jpg", self.edge_img, cmap="gray")


class HuffTransform(object):
    def __init__(self, edge_img, min_radius=150, max_radius=250) -> None:
        self.edge_img = edge_img
        self.sobel_x = np.array([[1, 0, -1], [2, 0, -2], [1, 0, -1]])
        self.sobel_y = np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]])
        self.gaussian_filter = np.array(
            [[0.10519, 0.11395, 0.10519], 
            [0.11395, 0.12344, 0.11395], 
            [0.10519, 0.11395, 0.10519]]
        )
        self.min_radius = min_radius
        self.max_radius = max_radius
        self.step = 2
        self.candidate = {}
        self.radius = {}
        self.edge_img_x = np.array([])
        self.edge_img_y = np.array([])
        self.par_space = np.array([])


    def calc_edge_theta(self):
        self.edge_img_x = conv_img(self.edge_img, self.sobel_x)
        self.edge_img_y = conv_img(self.edge_img, self.sobel_y)
        return 
    

    def find_center_coordination(self):
        a, b = self.edge_img.shape
        self.par_space = np.zeros(shape=(a, b))
        for i in range(1, a-2):
            for j in range(1, b-2):
                dx = np.sum(self.edge_img_x[i-1:i+2, j-1:j+2]*self.gaussian_filter)
                dy = np.sum(self.edge_img_y[i-1:i+2, j-1:j+2]*self.gaussian_filter)
                if self.edge_img[i, j] == 0 or (dx**2+dy**2 == 0):
                    continue
                theta_x = dx / np.sqrt(dx**2+dy**2)
                theta_y = dy / np.sqrt(dx**2+dy**2)
                for _ in range(2):
                    x = i + self.min_radius*theta_y
                    y = j + self.min_radius*theta_x
                    while max(i-self.max_radius, 1) < x < min(a-2, i+self.max_radius) and max(j-self.max_radius, 0) < y < min(b-2, j+self.max_radius):
                        self.par_space[int(x)-1:int(x)+1, int(y)-1:int(y)+1] += 1
                        x = x + self.step*theta_y
                        y = y + self.step*theta_x
                    theta_x = - theta_x
                    theta_y = - theta_y

        plt.imsave('./par_edge.jpg', self.par_space, cmap="gray")
        return 


    def determine_radius_coordination(self):
        while np.max(self.par_space) != 0:
            max_index, max_val = np.unravel_index(self.par_space.argmax(), self.par_space.shape), np.max(self.par_space)
            x, y = max_index
            flag = True
            if self.candidate:
                for index in self.candidate:
                    x_, y_ = index
                    if (x-x_)**2 + (y-y_)**2 < (2*self.min_radius)**2:
                        flag = False
                        break
            if flag:
                self.candidate[max_index] = max_val
            self.par_space[max(0, x-self.min_radius//2):min(a, x+self.min_radius//2), max(0, y-self.min_radius//2): min(b, y+self.min_radius//2)] = 0
        return


    def determine_radius(self):
        for cell in self.candidate:
            radius = np.zeros(shape=(self.max_radius+1, 1))
            x_i, y_i = cell
            for i in range(max(0, x_i-self.max_radius), min(a, x_i+self.max_radius)):
                for j in range(max(0, y_i-self.max_radius), min(b, y_i+self.max_radius)):
                    if self.edge_img[i, j] != 0 and not (self.edge_img_x[i, j] == 0  and self.edge_img_y[i, j] == 0):
                        r = np.sqrt((i-x_i)**2 + (j-y_i)**2)
                        if self.min_radius <= r <= self.max_radius:
                            radius[int(r)] += 1
            if np.max(radius) > 100:
                self.radius[cell] = (np.argmax(radius), np.max(radius))
        return 


    def plot_circle(self):
        for item in self.radius:
            a, b = item
            r = self.radius[item][0]
            self.edge_img[a-5:a+5, b-5:b+5] = 100
            for theta in np.arange(0, 2*np.pi, 0.1):
                x = int(a+r*np.cos(theta))
                y = int(b+r*np.sin(theta))
                self.edge_img[x-5:x+5, y-5:y+5] = 150
        plt.imsave('./HuffCircle.jpg', self.edge_img, cmap="gray")
        return 


class HarrisCorner(object):
    def __init__(self, img) -> None:
        self.img = img
        self.x, self.y = self.img.shape
        self.sobel_x = np.array([[1, 0, -1], [2, 0, -2], [1, 0, -1]])
        self.sobel_y = np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]])
        self.gaussian_template = np.array(
            [[0.0947416, 0.118318, 0.0947416], 
            [0.118318, 0.147761, 0.118318], 
            [0.0947416, 0.118318, 0.0947416]]
            )
        self.r = np.zeros(shape=(self.x, self.y, 3))

    
    def calc_r_matrix(self):
        conv_img_x = conv_img(self.img, self.sobel_x)
        conv_img_y = conv_img(self.img, self.sobel_y)
        Ix_sqr = conv_img(conv_img_x * conv_img_x, self.gaussian_template)
        Iy_sqr = conv_img(conv_img_y * conv_img_y, self.gaussian_template)
        Ixy_multi = conv_img(conv_img_x * conv_img_y, self.gaussian_template)
        for i in range(a):
            for j in range(b):
                mat = np.array([
                [Ix_sqr[i, j], Ixy_multi[i, j]], 
                [Ixy_multi[i, j], Iy_sqr[i, j]]
                ])
                val, _ = np.linalg.eig(mat)
                lam_1, lam_2 = val
                self.r[i, j, 0] = int(lam_1 * lam_2 - 0.04*((lam_1+lam_2)**2))
                self.r[i, j, 1] = lam_1
                self.r[i, j, 2] = lam_2


    def nms(self):
        copy_img = self.img.copy()
        threshold = np.max(self.r[:, :, 0]) * 0.01
        print("harris角点的门限值: ", threshold)
        for i in range(1, self.x-1):
            for j in range(1, self.y-1):
                cell = self.r[i-1:i+2, j-1:j+2, 0]
                if self.r[i, j, 0] == np.max(cell) and self.r[i, j, 0] > threshold:
                    copy_img[i, j] += 1000
        return copy_img


class SiftDetect(object):
    def __init__(self, img, s, min_size=4, sigma=1.6) -> None:
        self.img = img
        self.s = s
        self.min_size = min_size
        self.img_width = img.shape[0]
        self.img_heigth = img.shape[1]
        self.sigma = sigma
        self.min_length = min(self.img_width, self.img_heigth)
        self.octaves_num = int(np.log2(self.min_length) - self.min_size)
        self.dog_pyr = self.get_dog_pyramid(self.octaves_num, self.sigma, self.s, self.img)
        self.key_point_dic = self.get_key_points(self.octaves_num, self.dog_pyr, self.s+3)


    def get_dog_pyramid(self, octaves_num, sigma, s, img):
        images_num = s + 3
        k = np.power(2, 1/s)
        sigmas_list = [sigma] # 对应图像的sigma
        for i in range(1, images_num):
            tem = np.power(k, i) * sigma
            sigmas_list.append(np.sqrt(tem**2 - sigmas_list[i-1]**2))
        gaussian_scaled_pyramid = {}
        dog_pyramid = {}
        # 对每一组octave进行高斯处理
        for i in range(octaves_num):
            gaussian_scaled_pyramid[i] = []
            if i == 0:
                # 第一张图像不用动
                tem_img = img.copy()
            else:
                # 从第二组开始的图像都是前一组图像降采样得来的结果
                tem_img = tem_img[::2, ::2]
            cur_img = tem_img.copy()
            for j in range(images_num):
                cur_img = conv_img(cur_img, gaussian_template(sigmas_list[j]))
                gaussian_scaled_pyramid[i].append(cur_img.copy())
        # 构建差分金字塔
        for i in range(octaves_num):
            dog_pyramid[i] = []
            for j in range(1, images_num):
                dog_pyramid[i].append(gaussian_scaled_pyramid[i][j] - gaussian_scaled_pyramid[i][j-1])
        return dog_pyramid

    def get_main_direction(self, cor, bin_num, sigma, img):
        x, y = cor
        dx_list = []
        dy_list = []
        ori_list = []
        tem_hist = [0] * bin_num
        radius = int(4.5*sigma)
        k = 0
        for i in range(-radius, radius+1):
            tem_x = x + i
            if tem_x <= 0 or tem_x >= img.shape[0] - 1:
                continue
            for j in range(-radius, radius+1):
                tem_y = y + j
                if tem_y <= 0 or tem_y >= img.shape[1] - 1:
                    continue
                dx = img[tem_x, tem_y+1] - img[tem_x, tem_y-1]
                dy = img[tem_x+1, tem_y] - img[tem_x-1, tem_y]
                ori_list.append(np.arctan2(dy, dx)*180/np.pi)
                dx_list.append(dx)
                dy_list.append(dy)
                k += 1
        dx_list = np.array(dx_list)
        dy_list = np.array(dy_list)
        ori_list = np.array(ori_list)
        mag = (dx_list**2 + dy_list**2) ** 0.5
        for v in range(k):
            bin = int((bin_num / 360) * ori_list[v])
            if bin >= bin_num:
                bin -= bin_num
            if bin < 0:
                bin += bin_num
            tem_hist[bin] += mag[v]
        return np.argmax(tem_hist) * 10

    
    # 改进SIFT特征描述符（+3*sigma半径+线性插值）
    def calc_sift_descriptor(self, cor, sigma, direction, img):
        # 1. 计算半径r
        x, y = cor
        d = 4 # 四个区域
        bin_n = 8 # 分成八个方向，得到4*4*8
        radius = int(3 * sigma * np.sqrt(2) * (d+1) // 2)
        img_a, img_b = img.shape
        dx_list = [] # 储存x方向梯度
        dy_list = [] # 储存y方向梯度
        x_bin = [] # 储存x方向梯度值
        y_bin = [] # 储存y方向梯度值
        mag_list = [] # 储存梯度值
        ori_list = [] # 储存梯度方向
        hist = [0]*(d+2)*(d+2)*(bin_n+2) # 储存线性插值之后的值
        dst = [] # 储存最终描述子

        # 2. 计算主方向移动至正方向的旋转矩阵
        cos_t = np.cos(direction * (np.pi / 180)) / (3 * sigma) # 为了从radius变回bin里面的值
        sin_t = np.sin(direction * (np.pi / 180)) / (3 * sigma)

        # 3. 计算
        k = 0
        for i in range(-radius, radius+1):
            tem_x = x + i
            for j in range(-radius, radius+1):
                tem_y = y + j
                x_prime = i*cos_t - j*sin_t
                y_prime = i*sin_t + j*cos_t
                tem_x_bin = x_prime + d//2 - 0.5
                tem_y_bin = y_prime + d//2 - 0.5
                if -1 < tem_x_bin < d and -1 < tem_y_bin < d and 0 < tem_x < img_a-1 and 0 < tem_y < img_b-1:
                    dx = img[tem_x, tem_y+1] - img[tem_x, tem_y-1]
                    dy = img[tem_x+1, tem_y] - img[tem_x-1, tem_y]
                    dx_list.append(dx)
                    dy_list.append(dy)
                    x_bin.append(tem_x_bin)
                    y_bin.append(tem_y_bin)
                    k += 1

        # 4. 计算梯度及方向角
        length = k
        dx_list = np.array(dx_list)
        dy_list = np.array(dy_list)
        ori_list = np.arctan2(dy_list, dx_list) * 180 / np.pi
        mag_list = (dx_list ** 2 + dy_list ** 2) ** 0.5

        # 5. 三线性插值
        for k in range(length):
            tem_x_bin = x_bin[k]
            tem_y_bin = y_bin[k]
            tem_ori_bin = (ori_list[k]-direction) * bin_n / 360
            tem_mag = mag_list[k]
            x_0 = int(tem_x_bin)
            y_0 = int(tem_y_bin)
            o_0 = int(tem_ori_bin)
            tem_x_bin -= x_0
            tem_y_bin -= y_0
            tem_ori_bin -= o_0
            if o_0 < 0:
                o_0 += bin_n
            if o_0 >= bin_n:
                o_0 -= bin_n

            # 三线性插值
            v_r1 = tem_mag * tem_x_bin
            v_r0 = tem_mag - v_r1

            v_rc11 = v_r1 * tem_y_bin
            v_rc10 = v_r1 - v_rc11
            v_rc01 = v_r0 * tem_y_bin
            v_rc00 = v_r0 - v_rc01

            v_rco111 = v_rc11 * tem_ori_bin
            v_rco110 = v_rc11 - v_rco111
            v_rco101 = v_rc10 * tem_ori_bin
            v_rco100 = v_rc10 - v_rco101
            v_rco011 = v_rc01 * tem_ori_bin
            v_rco010 = v_rc01 - v_rco101
            v_rco001 = v_rc00 * tem_ori_bin
            v_rco000 = v_rc00 - v_rco001

            idx = ((x_0 + 1) * (d + 2) + y_0 + 1) * (bin_n + 2) + o_0
            
            hist[idx] += v_rco000
            hist[idx+1] += v_rco001
            hist[idx+(bin_n+2)] += v_rco010
            hist[idx+(bin_n+3)] += v_rco011
            hist[idx+(d+2)*(bin_n+2)] += v_rco100
            hist[idx+(d+2)*(bin_n+2)+1] += v_rco101
            hist[idx+(d+3)*(bin_n+2)] += v_rco110
            hist[idx+(d+3)*(bin_n+2)+1] += v_rco111

        # 最后完成d*d*n的统计
        for i in range(d):
            for j in range(d):
                idx = ((i+1)*(d+2) + (j+1)) * (bin_n+2)
                # 为了防止角度超过，漏算插值
                hist[idx] += hist[idx+bin_n]
                hist[idx+1] += hist[idx+bin_n+1]
                for k in range(bin_n):
                    dst.append(hist[idx+k])

        # 归一化，不是一定要做
        nrm_2 = 0
        length = d*d*bin_n
        for k in range(length):
            nrm_2 += dst[k] * dst[k]
        thr = np.sqrt(nrm_2) * 0.2

        nrm_2 = 0
        for k in range(length):
            val = min(dst[k], thr)
            dst[k] = val
            nrm_2 += val * val
        
        nrm_2 = 512 / np.sqrt(nrm_2)
        for k in range(length):
            dst[k] = min(max(dst[k]*nrm_2, 0), 255)
        return dst

    def get_key_points(self, octaves_num, dog_pyramid, images_num, sigma=1.6):
        dog_threshold = 0.06
        threshold = 0.5 * dog_threshold / 3 * 255
        key_points = {}
        gamma = 5
        for i in range(octaves_num):
            for j in range(1, images_num-2):
                down_img = dog_pyramid[i][j-1]
                mid_img = dog_pyramid[i][j]
                up_img = dog_pyramid[i][j+1]
                # 是否是局部极值，简单起见只考虑极大值
                row, col = mid_img.shape
                for k in range(1, row-1):
                    for v in range(1, col-1):
                        if mid_img[k, v] >= threshold and mid_img[k, v] == max(np.max(mid_img[k-1:k+2, v-1:v+2]), np.max(up_img[k-1:k+2, v-1:v+2]), np.max(down_img[k-1:k+2, v-1:v+2])):
                            # 子像素插值
                            max_iters = 5
                            convergence = False
                            k_, v_, j_ = k, v, j
                            for _ in range(max_iters):
                                if convergence:
                                    break
                                d_x = np.mat([(mid_img[k_][v_+1] - mid_img[k_][v_-1])*0.5, 0.5*(mid_img[k_+1][v_] - mid_img[k_-1][v_]), 0.5*(up_img[k_, v_]-down_img[k_, v_])]).T
                                d_xx = mid_img[k_][v_+1] + mid_img[k_][v_-1] - 2*mid_img[k_][v_]
                                d_yy = mid_img[k_+1][v_] + mid_img[k_-1][v_] - 2*mid_img[k_][v_]
                                d_sigma_2 = up_img[k_][v_] + down_img[k_][v_] - 2*mid_img[k_][v_]
                                d_xy = (mid_img[k_+1][v_+1] + mid_img[k_-1][v_-1] - mid_img[k_+1][v_-1] - mid_img[k_-1][v_+1]) * 0.25
                                d_x_sigma = (up_img[k_][v_+1] + down_img[k_][v_-1] - up_img[k_][v_-1] - down_img[k_][v_+1]) * 0.25
                                d_y_sigma = (up_img[k_-1][v_] + down_img[k_+1][v_] - up_img[k_+1][v_] - down_img[k_-1][v_]) * 0.25
                                dd_x = np.mat([[d_xx, d_xy, d_x_sigma], [d_xy, d_yy, d_y_sigma], [d_x_sigma, d_y_sigma, d_sigma_2]])
                                offset = - dd_x.I * d_x
                                if np.abs(offset[0]) < 0.5 and np.abs(offset[1]) < 0.5 and np.abs(offset[2]) < 0.5:
                                    response = mid_img[k_][v_] + 0.5 * d_x.T * offset
                                    if response > threshold:
                                        convergence = True
                                else:
                                    k_ += int(offset[0])
                                    v_ += int(offset[1])
                                    j_ += int(offset[2])
                                    if j_ < 1 or j_ > images_num - 2 or k_ < 1 or k_ >= row-1 or v_ < 1 or v_ >= col-1:
                                        break
                            if convergence:
                                trace = d_xx + d_yy
                                det = d_xx * d_yy - d_xy*d_xy
                                if det > 0 and (trace * trace) / det < (gamma+1)**2 / gamma:
                                    # 如果不用结构体，则使用元组记录所有信息
                                    k_ = k * np.power(2, i)
                                    v_ = v * np.power(2, i)
                                    size = sigma * np.power(2, i) * np.power(2, j/self.s)
                                    response = mid_img[k, v]
                                    # 计算主方向
                                    direction = self.get_main_direction((k, v), 36, size, mid_img)
                                    dst = self.calc_sift_descriptor((k, v), size, direction, mid_img)
                                    if not key_points.get((k_, v_)):
                                        key_points[(k_, v_)] = (size, response, direction, dst)
                                    else:
                                        _, res, _, _ = key_points[(k_, v_)]
                                        if res < response:
                                            key_points[(k_, v_)] = (size, response, direction, dst)
        return key_points


    def plot_sift_point(self):
        blob_img_ = self.img.copy()
        for cor in self.key_point_dic:
            x, y = cor
            size, _, _, _ = self.key_point_dic[cor]
            r = size * np.sqrt(2)
            for theta in np.arange(0, 2*np.pi, 0.1):
                blob_img_[int(x+r*np.cos(theta)), int(y+r*np.sin(theta))] += 255
        return blob_img_


def gaussian_template(sigma):
    width = int(6*sigma+1)
    if width % 2 == 0:
        width += 1
    gaussian = np.zeros(shape=(width, width))
    center = width >> 1
    # print(center)
    for i in range(width):
        for j in range(width):
            gaussian[i, j] = 1 / (2 * np.pi * sigma**2) * np.exp((-1 / (2*sigma**2)) * ((i-center)**2+(j-center)**2))
    s = np.sum(gaussian)
    # 归一化防止图像低于255
    gaussian = gaussian / s
    return gaussian


def conv_img(img, conv_template):
    a, b = img.shape
    a_, b_ = conv_template.shape
    a_ = a_ // 2
    b_ = b_ // 2
    img_ = np.zeros(shape=(a+2*a_, b+2*b_))
    img = np.pad(img, ((a_, a_), (b_, b_)), "constant", constant_values=(0, 0))
    for i in range(a_, a+a_):
        for j in range(b_, b+b_):
            cell = img[i-a_:i+a_+1, j-b_:j+b_+1]
            img_[i, j] = np.sum(cell * conv_template)
    return img_[a_:a+a_, b_:b+b_]


def edge_detect(img):
    canny = CannyEdge(img)
    # 1. 高斯滤波
    gaussian_img = canny.gaussian_filter()
    # 2. 边缘检测
    canny_edge = canny.edge_dect(gaussian_img)
    return canny_edge


def huff_circle(img):
    huff = HuffTransform(img)
    # 1. 计算边缘图梯度
    huff.calc_edge_theta()
    # 2. 确定可能的圆心坐标
    huff.find_center_coordination()
    # 3. 确定圆心坐标
    huff.determine_radius_coordination()
    # 4. 确定圆的半径
    huff.determine_radius()
    # 5. 将找出来的圆画出来
    huff.plot_circle()


def harris_corner(img):
    harris = HarrisCorner(img)
    # 1. 计算二阶矩矩阵及R值
    harris.calc_r_matrix()
    # 2. 非极大值抑制
    harris_img = harris.nms()
    # 3. 保存图
    plt.imsave('./harris.jpg', harris_img, cmap="gray")


def sift_detect(img, s=3, min_size=4, sigma=1.6):
    # 1. 检测sift特征点并得到特征点描述符
    sift = SiftDetect(img, s, min_size, sigma)
    # 2. 将检测出的sift特征点画在图上
    sift_img = sift.plot_sift_point()
    # 3. 保存图
    plt.imsave("./sift.jpg", sift_img, cmap="gray")
    return


if __name__ == "__main__":
    img = plt.imread(r"C:\Users\weitao\Desktop\Untitled Folder\cv\flower.jpg")
    img = np.array(img)
    a, b, c = img.shape
    new_img = np.zeros(shape=(a, b))
    print(new_img.shape)
    for i in range(a):
        for j in range(b):
            new_img[i, j] = img[i, j].mean()
    # 归一到0-255之间
    min_val = np.min(new_img)
    max_val = np.max(new_img)
    new_img = ((new_img - min_val) / (max_val - min_val)) * 255
    # canny_edge = edge_detect(new_img)
    # huff_circle(canny_edge)
    # harris_corner(new_img)
    sift_detect(new_img)
