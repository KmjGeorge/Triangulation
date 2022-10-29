# 3DRC
triangulation_direct(M1, M2, points1, points2, method)
    :param M1: 投影矩阵1
    :param M2: 投影矩阵2
    :param points1,  points2: 匹配点对的集合（欧式坐标）
    :param method: 求解方法 svd 奇异值分解  direct 对A^T·A作特征分解直接求解 
    return: 所有匹配对对应的P的欧式坐标

 triangulation_nonlinear(M1, M2, points1, points2, init_pws, method, verbose)
    :param M1: 投影矩阵1
    :param M2: 投影矩阵2
    :param points1,  points2: 匹配点对的集合（欧式坐标）
    :param init_pws: 世界坐标初始解(欧式坐标)
    :param method : 求解方法 newton 牛顿  BFGS 拟牛顿 gaussian 高斯牛顿  lm 列文伯格-马夸尔特
    :param verbose : 是否显示每次求解结果
    return: 所有匹配对对应的P的欧式坐标
 
