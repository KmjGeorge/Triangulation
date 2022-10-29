import cv2
import numpy as np
import matplotlib.pyplot as plt
import torch
from torch.autograd import Variable

# 内参数
alpha = 520.9
beta = 521.0
# theta = np.pi/2
cx = 325.1
cy = 249.7
K = np.array([alpha, 0, cx,
              0, 521.0, cy,
              0, 0, 1]).reshape([3, 3])


# 获取匹配点对
def get_points(img1, img2):
    # orb
    orb = cv2.ORB_create()
    kp1, des1 = orb.detectAndCompute(img1, None)
    kp2, des2 = orb.detectAndCompute(img2, None)
    # 特征匹配
    bf = cv2.BFMatcher(cv2.NORM_HAMMING)
    matches = bf.match(des1, des2)
    # 初步筛选
    min_distance = 10000
    max_distance = 0
    for x in matches:
        if x.distance < min_distance:
            min_distance = x.distance
        if x.distance > max_distance:
            max_distance = x.distance
    good_match = []
    for x in matches:
        if x.distance <= max(1.2 * min_distance, 20):
            good_match.append(x)
    print('匹配数：%d' % len(good_match))
    outimage = cv2.drawMatches(img1, kp1, img2, kp2, good_match, outImg=None)
    plt.figure(dpi=300)
    plt.imshow(cv2.cvtColor(outimage, cv2.COLOR_BGR2RGB))
    plt.axis('off')
    # plt.show()

    # 返回匹配点对
    points1 = []
    points2 = []
    for i in good_match:
        points1.append(list(kp1[i.queryIdx].pt))
        points2.append(list(kp2[i.trainIdx].pt))
    points1 = np.array(points1)
    points2 = np.array(points2)
    # print(points1[0].shape)  # 欧式坐标
    return points1, points2


# 计算一堆矩阵
def get_matrix(points1, points2):
    E, mask = cv2.findEssentialMat(points1, points2, K, cv2.RANSAC)  # 计算E
    _, R, T, _ = cv2.recoverPose(E, points1, points2, K, mask)  # 从E中分解出R和T
    M1 = np.matmul(K, np.array([[1, 0, 0, 0],  # 计算投影矩阵   M1 = K[I 0]  M2 = K[R T]
                                [0, 1, 0, 0],
                                [0, 0, 1, 0]]))
    M2 = np.matmul(K, np.concatenate((R, T), axis=1))
    return E, R, T, M1, M2


# 三角化 cv提供
def triangulation_cv(M1, M2, points1, points2):
    pointsW = cv2.triangulatePoints(M1, M2, points1.T, points2.T)  # 计算世界坐标
    # print(pointsW.shape)  # 4*n
    pointsW = (pointsW / pointsW[3]).T[:, :3]  # 转欧式坐标
    # print(pointsW.shape)  # n*3
    return pointsW


# 线性三角化
def triangulation_direct(M1, M2, points1, points2, method):
    """
    :param M1: 投影矩阵1
    :param M2: 投影矩阵2
    :param points1: 匹配点集1
    :param points2: 匹配点集2
    :param method: 求解方法 direct 直接求解  svd 奇异值分解
    :return: 所有匹配对对应的P的欧式坐标
    """
    pointsW = []
    # 分解M
    m11 = M1[0, :]
    m12 = M1[1, :]
    m13 = M1[2, :]
    m21 = M2[0, :]
    m22 = M2[1, :]
    m23 = M2[2, :]
    for p1, p2 in zip(points1, points2):
        u1 = p1[0]
        v1 = p1[1]
        u2 = p2[0]
        v2 = p2[1]
        # 构造矩阵A
        A = np.array([np.dot(u1, m13) - m11,
                      np.dot(v1, m13) - m12,
                      np.dot(u2, m23) - m21,
                      np.dot(v2, m23) - m22
                      ])
        if method == 'svd':
            _, _, V = np.linalg.svd(A, full_matrices=True, compute_uv=True)  # V 1*4
            # print(V)
            pointsW_euclid = (V[3] / V[3][3])[:3]  # 转欧式坐标
            # print(pointsW_euclid)
            pointsW.append(pointsW_euclid)  # V.T的最后一列即V的最后一行
        elif method == 'direct':
            # 构造矩阵A
            A = np.array([np.dot(u1, m13) - m11,
                          np.dot(v1, m13) - m12,
                          np.dot(u2, m23) - m21,
                          np.dot(v2, m23) - m22
                          ])
            # 对A^TA特征分解
            B = np.matmul(A.T, A)
            Lambda, V = np.linalg.eig(B)
            # 寻找最小特征值
            L_min_index = np.argmin(Lambda)
            # 对应的右特征向量
            V_min = V.T[L_min_index]
            pointsW_euclid = (V_min / V_min[3])[:3]  # 转欧式坐标
            pointsW.append(pointsW_euclid)  # V.T的最后一列即V的最后一行
        else:
            raise 'Error method type!'
    pointsW = np.array(pointsW)
    return pointsW


# 非线性三角化类
class NonLinear:
    def __init__(self, M1, M2, p1, p2):
        """
        :param M1: 投影矩阵1
        :param M2: 投影矩阵2
        :param p1: 匹配点1（欧式坐标）
        :param p2: 匹配点2（欧式坐标）
        """
        self.M1 = torch.Tensor(M1)
        self.M2 = torch.Tensor(M2)
        self.p1 = torch.Tensor(p1)
        self.p2 = torch.Tensor(p2)

    def loss(self, P):  # 计算loss = ||p-MP||² + ||p'-M'P||²
        """
        :param P：P的齐次坐标
        :return: P和理想投影点的距离平方
        """
        p1_proj = torch.matmul(self.M1, P)  # 在第一个摄像机投影后的P点
        p2_proj = torch.matmul(self.M2, P)  # 在第二个摄像机投影后的P点
        # print(p1_proj.shape) # 3*1

        # 投影点转欧式坐标
        p1_proj = torch.squeeze((p1_proj / p1_proj[2])[:2])  # (2,)
        p2_proj = torch.squeeze((p2_proj / p2_proj[2])[:2])  # (2,)
        y = torch.norm(self.p1 - p1_proj) ** 2 + torch.norm(self.p2 - p2_proj) ** 2
        return y

    def _loss(self, P):  # f1和f2
        """
        :param P：P的齐次坐标
        :return : d1和d2
        """
        p1_proj = torch.matmul(self.M1, P)
        p2_proj = torch.matmul(self.M2, P)
        p1_proj = torch.squeeze((p1_proj / p1_proj[2])[:2])  # (2,)
        p2_proj = torch.squeeze((p2_proj / p2_proj[2])[:2])  # (2,)
        y1 = torch.norm(self.p1 - p1_proj)
        y2 = torch.norm(self.p2 - p2_proj)
        return y1, y2

    def newton(self, P, epochs, err, ep, lr=1, verbose=False):
        """
        :param P: 需要求解的三维点P(欧式坐标)
        :param epochs: 最大迭代次数
        :param err: 迭代终止误差
        ;param ep: 迭代终止梯度
        :param lr: 学习率
        :param verbose : 打印计算结果
        :return: 求解结果(欧式坐标)
        """
        x = np.append(P, 1)  # 转齐次
        x = x[..., np.newaxis]  # 4*1
        x = Variable(torch.Tensor(x), requires_grad=True)
        loss_list = []
        epoch = 0
        while True:
            epoch += 1
            loss = self.loss(x)
            loss_list.append(loss.data)
            if epoch >= epochs:
                if verbose:
                    print('牛顿法已停止于: x=', end='')
                    print(x.data)
                    print('loss=', end='')
                    print(loss.data)
                    print('epochs=%d' % epoch)
                break
            if loss.data < err:  # 若误差小于err则结束迭代
                if verbose:
                    print('牛顿法已停止于: x=', end='')
                    print(x.data)
                    print('loss=', end='')
                    print(loss.data)
                    print('epochs=%d' % epoch)
                break
            grad = torch.autograd.grad(loss, x, create_graph=True, retain_graph=True)  # 计算梯度
            # print(grad[0])
            if torch.norm(grad[0], p=np.inf) <= ep:  # 若梯度范数小于某一阈值停止
                if verbose:
                    print('牛顿法已停止于: x=', end='')
                    print(x.data)
                    print('loss=', end='')
                    print(loss.data)
                    print('epochs=%d' % epoch)
                break
            # Hessian = torch.Tensor([])  # 计算Hessian矩阵
            # for item in grad[0]:
            #     Hessian = torch.cat((Hessian, torch.autograd.grad(item, x, retain_graph=True)[0]))
            # Hessian = Hessian.resize(4, 4)

            Hessian = torch.autograd.functional.hessian(self.loss, x)  # 4*1*4*1
            Hessian = torch.squeeze(Hessian)  # 4*4
            # print(Hessian.shape)
            try:
                Hessian_inverse = Hessian.inverse()  # 计算矩阵的逆
            except BaseException:
                raise 'Hessian矩阵是奇异的！'

            # 更新x
            delta_x = lr * torch.matmul(Hessian_inverse, grad[0])
            x = Variable(x.data - delta_x, requires_grad=True)

        return (x.data / x.data[3])[:3], loss_list

    def BFGS(self, P, epochs, err, ep, lr=1, verbose=False):
        """
        :param P: 需要求解的三维点P(欧式坐标)
        :param epochs: 最大迭代次数
        :param err: 迭代终止误差
        ;param ep: 迭代终止梯度
        :param lr: 学习率
        :param verbose : 打印计算结果
        :return: 求解结果(欧式坐标)
        """
        x = np.append(P, 1)  # 转齐次
        x = x[..., np.newaxis]  # 4*1
        x = Variable(torch.Tensor(x), requires_grad=True)  # 1*4
        loss_list = []
        Gk = torch.eye(4, 4)  # 初始Hessian矩阵的逆近似为单位矩阵
        epoch = 0
        while True:
            epoch += 1
            loss = self.loss(x)
            loss_list.append(loss.data)
            if epoch >= epochs:  # 第一次直接使用牛顿法
                if verbose:
                    print('BFGS法已停止于: x=', end='')
                    print(x.data)
                    print('loss=', end='')
                    print(loss.data)
                    print('epochs=%d' % epoch)
                break
            if loss.data < err:  # 若误差小于err则结束迭代
                if verbose:
                    print('BFGS法已停止于: x=', end='')
                    print(x.data)
                    print('loss=', end='')
                    print(loss.data)
                    print('epochs=%d' % epoch)
                break
            gk = torch.autograd.grad(loss, x, create_graph=True, retain_graph=True)[0]

            sk = - lr * torch.matmul(Gk, gk)
            # 更新x gk+1
            x = x + sk
            gk_new = torch.autograd.grad(self.loss(x), x, create_graph=True, retain_graph=True)[0]
            if torch.norm(gk_new, p=np.inf) <= ep:  # 若梯度范数小于某一阈值停止
                if verbose:
                    print('BFGS法已停止于: x=', end='')
                    print(x.data)
                    print('loss=', end='')
                    print(loss.data)
                    print('epochs=%d' % epoch)
                break
            yk = gk_new - gk

            # 更新Gk
            t1 = torch.eye(4) - torch.matmul(sk, yk.T) / torch.matmul(yk.T, sk)
            t2 = torch.eye(4) - torch.matmul(yk, sk.T) / torch.matmul(yk.T, sk)
            t3 = torch.matmul(sk, sk.T) / torch.matmul(yk.T, sk)
            if torch.isnan(t1).any() or torch.isnan(t2).any() or torch.isnan(t3).any():
                if verbose:
                    print('BFGS法已停止于: x=', end='')
                    print(x.data)
                    print('loss=', end='')
                    print(loss.data)
                    print('epochs=%d' % epoch)
                break
            Gk = torch.matmul(torch.matmul(t1, Gk), t2) + t3

        return (x.data / x.data[3])[:3], loss_list

    def gaussian(self, P, epochs, err, ep1, ep2, alpha=1.0, verbose=False):
        """
        :param P: 需要求解的三维点P(欧式坐标)
        :param epochs: 最大迭代次数
        :param err: 迭代终止误差
        :param ep1: epsilon1
        :param ep2: epsilon2
        :param alpha: 权值更新的步长权重
        :param verbose : 打印计算结果
        :return: 求解结果(欧式坐标)
        """

        x = np.append(P, 1)  # 转齐次
        x = x[..., np.newaxis]  # 4*1
        x = Variable(torch.Tensor(x), requires_grad=True)
        loss_list = []

        Jacobi = torch.autograd.functional.jacobian(self._loss, x)  # 计算J A g
        J = torch.Tensor([])
        for item in Jacobi:
            J = torch.cat((J, item))
        J = J.resize(2, 4)
        A = torch.matmul(J.T, J)  # 4*4

        f = torch.Tensor(self._loss(x))  # (2,)
        f.unsqueeze_(dim=1)  # (2,1)
        # for item in self._loss(x):
        #     f = torch.cat(f, item)
        g = torch.matmul(J.T, f)  # 4*1
        epoch = 0
        while True:
            epoch += 1
            loss = self.loss(x)
            loss_list.append(loss.data)
            if epoch >= epochs:
                if verbose:
                    print('Gaussian-newton法已停止于: x=', end='')
                    print(x.data)
                    print('loss=', end='')
                    print(loss.data)
                    print('epochs=%d' % epoch)
                break
            if loss.data < err:  # 若误差小于err则结束迭代
                if verbose:
                    print('Gaussian-newton法已停止于: x=', end='')
                    print(x.data)
                    print('loss=', end='')
                    print(loss.data)
                    print('epochs=%d' % epoch)
                break
            if torch.norm(g, p=np.inf) <= ep1:  # g的∞范数小于某一阈值停止迭代
                if verbose:
                    print('Gaussian-newton法已停止于: x=', end='')
                    print(x.data)
                    print('loss=', end='')
                    print(loss.data)
                    print('epochs=%d' % epoch)
                break

            # 解方程 Ah = -g
            # print(torch.linalg.matrix_rank(A))
            lu = torch.lu(A)
            h = torch.lu_solve(-g, *lu)
            if torch.norm(h) < ep2 * (torch.norm(x) + ep2):  # h小于某一阈值停止迭代
                if verbose:
                    print('Gaussian-newton法已停止于: x=', end='')
                    print(x.data)
                    print('loss=', end='')
                    print(loss.data)
                    print('epochs=%d' % epoch)
                break
            # 更新x J A g
            x = Variable(x.data + alpha * h, requires_grad=True)
            # tensor([[ 2.4725],
            #         [-3.0602],
            #         [16.3236],
            #         [ 1.0000]], requires_grad=True)

            # tensor([[  2.4930],
            #         [ -3.0609],
            #         [ 16.3235],
            #         [-23.9177]], requires_grad=True)

            Jacobi = torch.autograd.functional.jacobian(self._loss, x)  # 计算J A g
            J = torch.Tensor([])
            for item in Jacobi:
                J = torch.cat((J, item))
            J = J.resize(2, 4)
            A = torch.matmul(J.T, J)  # 4*4
            f = torch.Tensor(self._loss(x))  # (2,)
            f.unsqueeze_(dim=1)  # (2,1)
            g = torch.matmul(J.T, f)  # 4*1

        return (x.data / x.data[3])[:3], loss_list

    def lm(self, P, epochs, err, ep1=1e-8, ep2=1e-8, tau=1e-6, verbose=False):
        """
        :param P: 需要求解的三维点P(欧式坐标)
        :param epochs: 最大迭代次数
        :param err: 迭代终止误差
        :param ep1: epsilon1
        ;param ep2: epsilon2
        ;param tau: tau
        :param verbose : 打印计算结果
        :return: 求解结果(欧式坐标)
        """

        x = np.append(P, 1)  # 转齐次
        x = x[..., np.newaxis]  # 4*1
        x = Variable(torch.Tensor(x), requires_grad=True)
        loss_list = []
        # 初始化
        v = 2
        Jacobi = torch.autograd.functional.jacobian(self._loss, x)  # 计算J、A、g、μ
        J = torch.Tensor([])
        for item in Jacobi:
            J = torch.cat((J, item))
        J.resize_(2, 4)
        A = torch.matmul(J.T, J)
        f = torch.unsqueeze(torch.Tensor(self._loss(x)), dim=1)
        g = torch.matmul(J.T, f)
        max_aii = -np.inf
        for i in range(A.shape[0]):
            if max_aii < A[i][i]:
                max_aii = A[i][i]
        mu = tau * max_aii
        # print(mu)
        epoch = 0
        while True:
            epoch += 1
            loss = self.loss(x)
            loss_list.append(loss.data)
            if epoch >= epochs:
                if verbose:
                    print('L-M法已停止于: x=', end='')
                    print(x.data)
                    print('loss=', end='')
                    print(loss.data)
                    print('epochs=%d' % epoch)
                break

            if loss.data < err:  # 若误差小于err则结束迭代
                if verbose:
                    print('L-M法已停止于: x=', end='')
                    print(x.data)
                    print('loss=', end='')
                    print(loss.data)
                    print('epochs=%d' % epoch)
                break
            if torch.norm(g, p=np.inf) <= ep1:  # g的∞范数小于某一阈值停止迭代
                if verbose:
                    print('L-M法已停止于: x=', end='')
                    print(x.data)
                    print('loss=', end='')
                    print(loss.data)
                    print('epochs=%d' % epoch)
                break

            # 解方程 (A+μI)h = -g
            lu = torch.lu(A + mu * torch.eye(A.shape[0], A.shape[1]))   # A+μI是满秩的
            h = torch.lu_solve(-g, *lu)  # 4*1
            if torch.norm(h) < ep2 * (torch.norm(x) + ep2):  # h小于某一阈值停止迭代
                if verbose:
                    print('L-M法已停止于: x=', end='')
                    print(x.data)
                    print('loss=', end='')
                    print(loss.data)
                    print('epochs=%d' % epoch)
                break

            x_new = x.data + h
            #  计算rho
            rho = (loss - self.loss(x_new)) / 0.5 * (torch.matmul(h.T, (mu * h - g)))
            if rho > 0:
                # 更新
                x = Variable(x_new, requires_grad=True)
                Jacobi = torch.autograd.functional.jacobian(self._loss, x)  # 计算J、A、g、μ
                J = torch.Tensor([])
                for item in Jacobi:
                    J = torch.cat((J, item))
                J.resize_(2, 4)
                A = torch.matmul(J.T, J)
                f = torch.unsqueeze(torch.Tensor(self._loss(x)), dim=1)
                g = torch.matmul(J.T, f)
                mu = mu * max(1 / 3, 1 - (2 * rho - 1) ** 3)
                v = 2
            else:
                mu = mu * v
                v = 2 * v

        return (x.data / x.data[3])[:3], loss_list


# 非线性三角化
def triangulation_nonlinear(M1, M2, points1, points2, init_pws, method, verbose=False):
    """
    :param M1: 投影矩阵1
    :param M2: 投影矩阵2
    :param points1, points2: 匹配点对的集合（欧式坐标
    :param init_pws: 世界坐标初始解(欧式坐标)
    :param method : 求解方法 newton 牛顿 gaussian 高斯牛顿 lm 列文伯格-马夸尔特
    :param verbose : 显示求解结果
    :return:
    """
    pws = []
    for p1, p2, pw in zip(points1, points2, init_pws):
        solver = NonLinear(M1, M2, p1, p2)
        if method == 'newton':
            pw2, loss_lst = solver.newton(pw, epochs=500, err=1e-3, ep=1e-6, lr=1,  verbose=verbose)
        elif method == 'BFGS':
            pw2, loss_lst = solver.BFGS(pw, epochs=500, err=1e-3, ep=1e-6, lr=1, verbose=verbose)
        elif method == 'lm':
            pw2, loss_lst = solver.lm(pw, epochs=500, err=1e-3, ep1=1e-8, ep2=1e-8, tau=1e-3, verbose=verbose)
        elif method == 'gaussian':
            pw2, loss_lst = solver.gaussian(pw, epochs=500, err=1e-3, ep1=1e-6, ep2=1e-6, alpha=1.0, verbose=verbose)

        else:
            raise 'Error method type!'
        pw2 = np.squeeze(np.array(pw2))  # tensor(3,1)转ndarray(3,)
        pws.append(pw2)

        # # 绘制loss曲线
        # if len(loss_lst):
        #     epochs = [i + 1 for i in range(len(loss_lst))]
        #     plt.plot(epochs, loss_lst)
        #     length = max(loss_lst) - min(loss_lst)
        #     plt.ylim(loss_lst[0] - length, loss_lst[0] + length)
        #     plt.show()
    pws = np.array(pws)
    return pws


def get_color(z):
    maxz = 50
    minz = 10
    range = maxz - minz
    if z > maxz:
        z = maxz
    if z < minz:
        z = minz
    return 255 * z / range, 0, 255 * (1 - z / range)


# 可视化
def show_triangulation(img1, img2, points1, points2, pointsW, R, T):
    for i in range(points1.shape[0]):  # 为每个三角化完成的点绘制一个圆圈，颜色代表深度
        # 图一
        img1 = cv2.circle(img1, (points1[i][0].astype(int), points1[i][1].astype(int)), 10, get_color(pointsW[i, 2]),
                          -1)
        # 图二
        tmp_point = np.dot(R, pointsW[i, :].reshape(3, 1)) + T
        tmp_point = tmp_point.reshape(-1)
        img2 = cv2.circle(img2, (points2[i][0].astype(int), points2[i][1].astype(int)), 10, get_color(tmp_point[2]), -1)
    fig = plt.figure(dpi=300)
    fig.add_subplot(121)
    plt.imshow(img1[:, :, ::-1])
    plt.axis('off')
    fig.add_subplot(122)
    plt.imshow(img2[:, :, ::-1])
    plt.axis('off')
    plt.show()


# 使用例
if __name__ == '__main__':
    img_path1 = 'images/teddy/im2.png'
    img_path2 = 'images/teddy/im6.png'

    # 准备好匹配点对和投影矩阵
    img1 = cv2.imread(img_path1)
    img2 = cv2.imread(img_path2)
    ps1, ps2 = get_points(img1, img2)
    _, R, T, M1, M2 = get_matrix(ps1, ps2)

    # 线性三角化
    """
    triangulation_direct(M1, M2, points1, points2, method)
    :param M1: 投影矩阵1
    :param M2: 投影矩阵2
    :param points1,  points2: 匹配点对的集合（欧式坐标）
    :param method: 求解方法 svd 奇异值分解  direct 对A^T·A作特征分解直接求解 
    :param return: 所有匹配对对应的P的欧式坐标
    """
    pws1 = triangulation_direct(M1, M2, ps1, ps2, method='direct')
    # pws2 = triangulation_direct(M1, M2, ps1, ps2, method='svd')

    # 非线性三角化
    """
    triangulation_nonlinear(M1, M2, points1, points2, init_pws, method, verbose)
    :param M1: 投影矩阵1
    :param M2: 投影矩阵2
    :param points1,  points2: 匹配点对的集合（欧式坐标）
    :param init_pws: 世界坐标初始解(欧式坐标)
    :param method : 求解方法 newton 牛顿  BFGS 拟牛顿 gaussian 高斯牛顿  lm 列文伯格-马夸尔特
    :param verbose : 是否显示每次求解结果
    return: 所有匹配对对应的P的欧式坐标
    """
    # pws3 = triangulation_nonlinear(M1, M2, ps1, ps2, init_pws=pws1, method='newton', verbose=False)  # 将线性解作为非线性法的初始解
    # pws4 = triangulation_nonlinear(M1, M2, ps1, ps2, init_pws=pws1, method='BFGS', verbose=False)
    # pws6 = triangulation_nonlinear(M1, M2, ps1, ps2, init_pws=pws1, method='gaussian', verbose=False)
    pws5 = triangulation_nonlinear(M1, M2, ps1, ps2, init_pws=pws1, method='lm', verbose=False)
    print(pws1)
    print('=============================================')
    print(pws5)


