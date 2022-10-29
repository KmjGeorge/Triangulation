import cv2
import numpy as np

img = cv2.imread('images/100001.jpg', cv2.IMREAD_GRAYSCALE)
img = cv2.resize(img, (408, 544))  # (h, w, c)    (816, 612)

width = img.shape[0]
height = img.shape[1]
canvas_width = 1200
canvas_height = 1500
canvas = np.zeros((canvas_width, canvas_height), dtype=np.uint8)  # 画布

# 画布上img的原点坐标（左上角）
dx = int((canvas_width - width) / 2) - 1  # x_img = x_canvas + dx
dy = int((canvas_height - height) / 2) - 1  # y_img = y_canvas + dy


# 获取每个点的欧式坐标（相对于画布）
def get_position_euclid():
    position_list = []
    for i in range(width):
        for j in range(height):
            p = (i + dx, j + dy)
            position_list.append(p)
    # print(position_list)
    return np.array(position_list).reshape((width, height, 2))


# 欧式坐标转齐次坐标
def eucild_to_homo(position_euclid, w):
    position_list = []
    for i in range(width):
        for j in range(height):
            x = position_euclid[i][j][0]
            y = position_euclid[i][j][1]
            p = (x, y, w)
            position_list.append(p)
    return np.array(position_list).reshape((width, height, 3))


def homo_to_euclid(position_homo):
    position_list = []
    for i in range(width):
        for j in range(height):
            x = position_homo[i][j][0]
            y = position_homo[i][j][1]
            w = position_homo[i][j][2]
            p = (int(x / w), int(y / w))
            position_list.append(p)
    return np.array(position_list).reshape((width, height, 2))


# 对齐次空间的点作欧式变换
def euclid_trans(position_homo, sigma, theta, x0, y0):
    martix = np.zeros(shape=(3, 3))
    martix[0][0], martix[0][1], martix[0][2] = sigma * np.cos(theta), -np.sin(theta), x0
    martix[1][0], martix[1][1], martix[1][2] = sigma * np.sin(theta), np.cos(theta), y0
    martix[2][0], martix[2][1], martix[2][2] = 0, 0, 1

    position_homo_trans = []
    for i in range(width):
        for j in range(height):
            position_homo_trans.append(np.matmul(martix, position_homo[i][j]))
    return np.array(position_homo_trans).reshape((width, height, 3))


# 对齐次空间的点作相似变换
def similar_trans(position_homo, s, theta, x0, y0):
    martix = np.zeros(shape=(3, 3))
    martix[0][0], martix[0][1], martix[0][2] = s * np.cos(theta), -s * np.sin(theta), x0
    martix[1][0], martix[1][1], martix[1][2] = s * np.sin(theta), s * np.cos(theta), y0
    martix[2][0], martix[2][1], martix[2][2] = 0, 0, 1

    position_homo_trans = []
    for i in range(width):
        for j in range(height):
            position_homo_trans.append(np.matmul(martix, position_homo[i][j]))
    return np.array(position_homo_trans).reshape((width, height, 3))


# 对齐次空间的点作仿射变换
def affine_trans(position_homo, a, b, c, d, x0, y0):
    martix = np.zeros(shape=(3, 3))
    martix[0][0], martix[0][1], martix[0][2] = a, b, x0
    martix[1][0], martix[1][1], martix[1][2] = c, d, y0
    martix[2][0], martix[2][1], martix[2][2] = 0, 0, 1

    position_homo_trans = []
    for i in range(width):
        for j in range(height):
            position_homo_trans.append(np.matmul(martix, position_homo[i][j]))
    return np.array(position_homo_trans).reshape((width, height, 3))


# 对齐次空间的点作透视变换
def perspective_trans(position_homo, a, b, c, d, v1, v2, x0, y0):
    martix = np.zeros(shape=(3, 3))
    martix[0][0], martix[0][1], martix[0][2] = a, b, x0
    martix[1][0], martix[1][1], martix[1][2] = c, d, y0
    martix[2][0], martix[2][1], martix[2][2] = v1, v2, 1
    # print(martix)

    position_homo_trans = []
    for i in range(width):
        for j in range(height):
            position_homo_trans.append(np.matmul(martix, position_homo[i][j]))
    return np.array(position_homo_trans).reshape((width, height, 3))


# 画图
def draw(position):
    for i in range(width):
        for j in range(height):
            x = position[i][j][0]
            y = position[i][j][1]
            canvas[x][y] = img[i][j]
    # print(canvas.shape)
    # print(img.shape)
    cv2.imshow("image", canvas)
    cv2.waitKey(0)


# 清空画布
def flush_canvans():
    for i in range(canvas_width):
        for j in range(canvas_height):
            canvas[i][j] = 0


if __name__ == '__main__':
    pos_eucild = get_position_euclid()
    draw(pos_eucild)
    flush_canvans()

    # 变为齐次空间
    pos_homo = eucild_to_homo(pos_eucild, w=1)

    '''欧式变换'''
    # pos_homo_trans = euclid_trans(position_homo=pos_homo, sigma=1, theta=np.pi/2, x0=200, y0=200)
    # pos_eucild_trans = homo_to_euclid(pos_homo_trans)
    '''相似变换'''
    # pos_homo_trans = similar_trans(position_homo=pos_homo, s=0.5, theta=-np.pi / 2, x0=0, y0=0)
    # pos_eucild_trans = homo_to_euclid(pos_homo_trans)
    '''仿射变换'''
    # pos_homo_trans = affine_trans(position_homo=pos_homo, a=0.4, b=0.8, c=0.7, d=0.1, x0=-200, y0=100)
    # pos_eucild_trans = homo_to_euclid(pos_homo_trans)
    '''透视变换'''
    pos_homo_trans = perspective_trans(position_homo=pos_homo, a=1.2, b=0.6, c=0.4, d=1.5, v1=0.0005, v2=0.0001,
                                       x0=-500, y0=-300)

    # 变回欧式空间
    pos_eucild_trans = homo_to_euclid(pos_homo_trans)

    draw(pos_eucild_trans)
