import numpy as np
import matplotlib.pyplot as plt

def loadDataSet(filename):
    dataMat, labelMat = [], []
    with open(filename) as fr:
        for line in fr.readlines():
            lineArr = []
            curLine = line.strip().split('\t')
            for i in range(len(curLine)-1):
                lineArr.append(float(curLine[i]))
            dataMat.append(lineArr)
            labelMat.append(float(curLine[-1]))
        return dataMat, labelMat

def standRegres(xArr, yArr):
    """
    最小二乘法计算回归系数
    :param xArr: list[list], 特征数据
    :param yArr: list, 回归数据
    :return:
        - 回归系数
    """
    xMat = np.mat(xArr)
    yMat = np.mat(yArr).T
    xTx = xMat.T*xMat
    if np.linalg.det(xTx) == 0.0:
        print("This matrix is singular, cannot do inverse")
        return
    ws = xTx.I * (xMat.T*yMat)
    return ws

def lwlr(testPoint, xArr, yArr, k=1.):
    """
    局部加权
    :param testPoint:
    :param xArr:
    :param yArr:
    :param k:
    :return:
    """
    xMat = np.mat(xArr)
    yMat = np.mat(yArr).T
    m = np.shape(xMat)[0]
    weights = np.mat(np.eye((m)))
    for j in range(m):
        diffMat = testPoint - xMat[j, :]
        weights[j, j] = np.exp(diffMat * diffMat.T/(-2.*k**2))
    xTx = xMat.T * (weights * xMat)
    if np.linalg.det(xTx) == 0.:
        print("this matrix is singular, cannot do inverse")
        return
    ws = xTx.I * (xMat.T * (weights * yMat))
    return testPoint * ws

def lwlrTest(testArr, xArr, yArr, k=1.):
    m = np.shape(testArr)[0]
    yHat = np.zeros(m)
    for i in range(m):
        yHat[i] = lwlr(testArr[i], xArr, yArr, k)
    return yHat

def ridgeRegress(xMat, yMat, lam=0.2):
    """
    模型缩减： 岭回归
    :param xMat:
    :param yMat:
    :param lam:
    :return:
    """
    xTx = xMat.T*xMat
    denom = xTx + np.eye(xMat.shape[1])*lam
    if np.linalg.det(denom) == 0.:
        print("this matrix is singular, cannot do inverse")
        return
    ws = denom.I * (xMat.T*yMat)
    return ws

def ridgeTest(xArr, yArr):
    xMat = np.mat(xArr)
    yMat = np.mat(yArr).T
    yMean = np.mean(yMat, 0)
    yMat = yMat-yMean
    xMeans = np.mean(xMat, 0)
    xVar = np.var(xMat, 0)
    xMat = (xMat-xMeans)/xVar
    numTestPts = 30
    wMat = np.zeros((numTestPts, xMat.shape[1]))
    for i in range(numTestPts):
        ws = ridgeRegress(xMat, yMat, np.exp(i-10))
        wMat[i, :] = ws.T
    return wMat


if __name__ == "__main__":
    xArr, yArr = loadDataSet('./data/ex0.txt')
    # ws = standRegres(xArr, yArr)
    # xMat = np.mat(xArr)
    # yMat = np.mat(yArr)
    # yHat = xMat*ws
    # fig = plt.figure()
    # ax = fig.add_subplot(111)
    # ax.scatter(xMat[:, 1].flatten().A[0], yMat.T[:, 0].flatten().A[0])
    # xCopy = xMat.copy()
    # xCopy.sort(0)
    # yHat = xCopy*ws
    # ax.plot(xCopy[:, 1], yHat)
    # plt.show()
    # corr = np.corrcoef(yHat.T, yMat)
    # print(corr)

    # yHat = lwlrTest(xArr, xArr, yArr, 0.003)
    # xMat = np.mat(xArr)
    # srtInd = xMat[:, 1].argsort(0)
    # xSort = xMat[srtInd][:,0,:]
    # fig = plt.figure()
    # ax = fig.add_subplot(111)
    # ax.plot(xSort[:,1], yHat[srtInd])
    # ax.scatter(xMat[:, 1].flatten().A[0], np.mat(yArr).T.flatten().A[0], s=2)
    # plt.show()

    abX, abY = loadDataSet('./data/abalone.txt')
    ridgeWeights = ridgeTest(abX, abY)
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.plot(ridgeWeights)
    plt.show()