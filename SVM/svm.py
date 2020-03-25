import random
import numpy as np
import matplotlib.pyplot as plt
from SVM import platt_smo
"""
SMO算法实现
"""

"""
loadDataSet:加载数据

:param
    filename - 文件路径
    
:return
    dataMat - list[list], 数据
    labelMat - list, 标签
"""
def loadDataSet(fileName):
    dataMat, labelMat = [], []
    with open(fileName) as fr:
        for line in fr.readlines():
            lineArr = line.strip().split('\t')
            dataMat.append([float(lineArr[0]), float(lineArr[1])])
            labelMat.append(float(lineArr[2]))
        return dataMat, labelMat

"""
selectJrand: 随机选择alpha(j)

:param
    i - 选择的alpha(i)的编号
    m - 样本数目
    
:return 
    j - alpha(j)的编号
"""
def selectJrand(i, m):
    j = i
    while j == i:
        j = int(random.uniform(0, m))
    return j

"""
clipAlpha: 对Alpha进行裁剪, L <= aj <= H
"""
def clipAlpha(aj, H, L):
    if aj > H:
        aj = H
    if L > aj:
        aj = L
    return aj


def smoSimple(dataMatIn, classLabels, C, toler, maxIter):
    """
    SMO算法: 迭代更新alpha, 每次选择一对alpha进行更新, 如果后面不能对alpha对进行有效更新, 则算法执行完毕
    :param dataMatIn: 输入数据
    :param classLabels: 标签
    :param C: 惩罚参数
    :param toler: ？？
    :param maxIter: 最大的迭代次数
    :return:
        alphas, b
    """
    dataMatrix, labelMat = np.mat(dataMatIn), np.mat(classLabels).transpose()
    b = 0
    # m - 样本个数, n - 特征个数
    m, n = np.shape(dataMatrix)
    alphas = np.mat(np.zeros((m, 1)))
    iter = 0
    while iter < maxIter:
        alphaPairsChanged = 0
        for i in range(m):
            fxi = float(np.multiply(alphas, labelMat).T * (dataMatrix*dataMatrix[i, :].T)) + b
            Ei = fxi - float(labelMat[i])
            # 这一句感觉不是太懂
            if ((labelMat[i]*Ei < -toler) and (alphas[i] < C)) or ((labelMat[i]*Ei > toler) and (alphas[i] > 0)):
                j = selectJrand(i, m)
                fxj = float(np.multiply(alphas, labelMat).T * (dataMatrix*dataMatrix[j, :].T)) + b
                Ej = fxj - float(labelMat[j])
                # 这里应该不需要使用copy
                alphaIold = alphas[i].copy()
                alphaJold = alphas[j].copy()
                # 确定L和H
                if labelMat[i] != labelMat[j]:
                    L = max(0, alphas[j]-alphas[i])
                    H = min(C, C+alphas[j]-alphas[i])
                else:
                    L = max(0, alphas[j]+alphas[i]-C)
                    H = min(C, alphas[j]+alphas[i])
                if L == H:
                    print("L==H")
                    continue
                eta = 2.0*dataMatrix[i, :]*dataMatrix[j, :].T - dataMatrix[i, :]*dataMatrix[i, :].T - dataMatrix[j, :]*dataMatrix[j, :].T
                if eta >= 0:
                    print("eta >= 0")
                    continue
                alphas[j] -= labelMat[j]*(Ei - Ej)/eta
                alphas[j] = clipAlpha(alphas[j], H, L)
                if abs(alphas[j]-alphaJold) < 0.00001:
                    print("j not moving enough")
                    continue
                alphas[i] += labelMat[j]*labelMat[i]*(alphaJold - alphas[j])
                b1 = b-Ei-labelMat[i]*(alphas[i]-alphaIold)*(dataMatrix[i, :]*dataMatrix[i, :].T) - labelMat[j]*(alphas[j]-alphaJold)*(dataMatrix[i, :]*dataMatrix[j, :].T)
                b2 = b-Ej-labelMat[i]*(alphas[i]-alphaIold)*(dataMatrix[i, :]*dataMatrix[i, :].T) - labelMat[j]*(alphas[j]-alphaJold)*(dataMatrix[i, :]*dataMatrix[j, :].T)
                if 0 < alphas[i] and C > alphas[i]:
                    b = b1
                elif 0 < alphas[j] and C > alphas[j]:
                    b = b2
                else:
                    b = (b1+b2)/2.0
                alphaPairsChanged += 1
                print("iter: {} i: {}, pairs changed {}".format(iter, i, alphaPairsChanged))
        if alphaPairsChanged == 0:
            iter += 1
        else:
            iter = 0
        print("iteration number: {}".format(iter))
    return b, alphas

def smoP(dataMatIn, classLabels, C, toler, maxIter):
    oS = platt_smo.optStruct(np.mat(dataMatIn), np.mat(classLabels).transpose(), C, toler)
    iter = 0
    entireSet, alphaPairsChanged = True, 0
    while (iter < maxIter) and ((alphaPairsChanged > 0) or entireSet):
        alphaPairsChanged = 0
        if entireSet:
            for i in range(oS.m):
                alphaPairsChanged += oS.innerL(i)
            print("fullSet, iter: {} i: {}, pairs changed {}".format(iter, i, alphaPairsChanged))
            iter += 1
        else:
            # 遍历所有0 < alphas < C, 即所有在支持向量上的点
            nonBoundIs = np.nonzero((oS.alphas.A > 0)*(oS.alphas.A < C))[0]
            for i in nonBoundIs:
                alphaPairsChanged += oS.innerL(i)
                print("non-bound, iter: {} i: {}, pairs changed {}".format(iter, i, alphaPairsChanged))
            iter += 1
        if entireSet:
            entireSet = False
        elif alphaPairsChanged == 0:
            entireSet = True
        print("iteration number: {}".format(iter))
    return oS.b, oS.alphas


def calcWs(alphas, dataArr, classLabels):
    X = np.mat(dataArr)
    labelMat = np.mat(classLabels).transpose()
    m, n = np.shape(X)
    w = np.zeros((n, 1))
    for i in range(m):
        w += np.multiply(alphas[i]*labelMat[i], X[i, :].T)
    return w


def myplot(datas, labels):
    import matplotlib.pyplot as plt
    x = list(zip(*datas))
    labels = [(x+3)*20 for x in labels]
    # print(labels)
    # 颜色为负数貌似就是白色
    plt.scatter(list(x[0]), list(x[1]), c=labels)
    # plt.show()

if __name__ == '__main__':
    dataArr, labelArr = loadDataSet('f:\\machine-learning\\data\\SVM\\testSet.txt')
    # print(labelArr)
    b, alphas = smoP(dataArr, labelArr, 0.6, 0.001, 40)
    # print(b)
    myplot(dataArr, labelArr)
    data = []
    # 获取支持向量
    for i in range(100):
        if alphas[i] > 0:
            print(dataArr[i])
            print(labelArr[i])
            data.append(dataArr[i])
    x = list(zip(*data))
    plt.scatter(list(x[0]), list(x[1]), s=100, marker='x')
    plt.show()

    ws = calcWs(alphas, dataArr, labelArr)
    print(ws)

    # test
    t = np.array([[10, 0]])
    ans = np.dot(t, ws)+b
    print(ans)

