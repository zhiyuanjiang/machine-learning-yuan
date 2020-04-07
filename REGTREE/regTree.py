import numpy as np

def loadDataSet(filename):
    dataMat = []
    with open(filename, 'r') as fr:
        for line in fr.readlines():
            curLine = line.strip().split('\t')
            fltLine = map(float, curLine)
            dataMat.append(list(fltLine))
        return dataMat

def binSplitDataSet(dataSet, feature, value):
    """
    对数据集进行二元切分
    :param dataSet: 数据集
    :param feature: 特征下标
    :param value: 阈值
    :return:
        - 两个切分的数据集
    """
    mat0 = dataSet[np.nonzero(dataSet[:, feature] > value), :][0]
    mat1 = dataSet[np.nonzero(dataSet[:, feature] <= value), :][0]
    return mat0, mat1

def regLeaf(dataSet):
    return np.mean(dataSet[:, -1])

def regErr(dataSet):
    return np.var(dataSet[:, -1]) * dataSet.shape[0]

def createTree(dataSet, leafType=regLeaf, errType=regErr, ops=(1,4)):
    """
    创建一棵回归树
    :param dataSet: 数据集
    :param leafType: 计算叶子节点值的函数
    :param errType: 计算节点误差的函数
    :param ops: 传入的超参数
    :return:
        - 返回一棵dict类型的回归树
    """
    feat, val = chooseBestSplit(dataSet, leafType, errType, ops)
    if feat == None:
        return val
    retTree = {}
    retTree['spInd'] = feat
    retTree['spVal'] = val
    lSet, rSet = binSplitDataSet(dataSet, feat, val)
    retTree['left'] = createTree(lSet, leafType, errType, ops)
    retTree['right'] = createTree(rSet, leafType, errType, ops)
    return retTree


def chooseBestSplit(dataSet, leafType=regLeaf, errType=regErr, ops=(1,4)):
    # 选择对于一个节点最佳的划分特征和最佳的划分值

    tolS, tolN = ops[0], ops[1]
    if len(set(dataSet[:, -1].tolist())) == 1:
        return None, leafType(dataSet)
    m, n = dataSet.shape
    S = errType(dataSet)
    bestS = np.inf
    bestIndex = 0
    bestValue = 0
    for featIndex in range(n-1):
        for splitVal in set(dataSet[:, featIndex]):
            mat0, mat1 = binSplitDataSet(dataSet, featIndex, splitVal)
            if mat0.shape[0] < tolN or mat1.shape[0] < tolN:
                continue
            newS = errType(mat0)+errType(mat1)
            if newS < bestS:
                bestIndex = featIndex
                bestValue = splitVal
                bestS = newS
    if S-bestS < tolS:
        return None, leafType(dataSet)
    mat0, mat1 = binSplitDataSet(dataSet, bestIndex, bestValue)
    if mat0.shape[0] < tolN or mat1.shape[0] < tolN:
        return None, leafType(dataSet)
    return bestIndex, bestValue

def isTree(obj):
     # 判断obj是否是一棵树(是不是一个dict对象)
     return (type(obj).__name__ == 'dict')

def getMean(tree):
    # 计算一棵树的叶子节点的"平均值"
    if isTree(tree['right']):
        tree['right'] = getMean(tree['right'])
    if isTree(tree['left']):
        tree['left'] = getMean(tree['left'])
    return (tree['left']+tree['right'])/2.

def prune(tree, testData):
    # 如果没有数据在这课树上，将这棵树变成一个叶子节点
    if testData.shape[0] == 0:
        return getMean(tree)
    if isTree(tree['right']) or isTree(tree['left']):
        lSet, rSet = binSplitDataSet(testData, tree['spInd'], tree['spVal'])
    if isTree(tree['left']):
        tree['left'] = prune(tree['left'], lSet)
    if isTree(tree['right']):
        tree['right'] = prune(tree['right'], rSet)
    # 如果右子树和左子树通过剪枝都变成了一个叶子节点，计算误差看能不能继续合并。
    if not isTree(tree['left']) and not isTree(tree['right']):
        lSet, rSet = binSplitDataSet(testData, tree['spInd'], tree['spVal'])
        errorNoMerge = np.sum(np.power(lSet[:,-1]-tree['left'], 2)) + np.sum(np.power(rSet[:,-1]-tree['right'], 2))
        treeMean = (tree['left']+tree['right'])/2.
        errorMerge = np.sum(np.power(testData[:,-1]-treeMean, 2))
        if errorMerge < errorNoMerge:
            print("merging")
            return treeMean
        else:
            return tree
    else:
        return tree

def linearSolve(dataSet):
    # 对数据进行线性回归
    dataSet = np.mat(dataSet)
    m, n = dataSet.shape
    X = np.mat(np.ones((m, n)))
    Y = np.mat(np.ones((m, 1)))
    X[:, 1:n] = dataSet[:, 0:n-1]
    Y = dataSet[:, -1]
    xTx = X.T*X
    if np.linalg.det(xTx) == 0.:
        raise  NameError('This matrix is singular, cannot do inverse, try increasing the second value of ops')
    ws = xTx.I * (X.T * Y)
    return ws.A, X.A, Y.A

def modelLeaf(dataSet):
    # 生成模型树的叶节点
    ws, X, Y = linearSolve(dataSet)
    return ws

def modelErr(dataSet):
    # 误差计算
    ws, X, Y = linearSolve(dataSet)
    yHat = np.dot(X, ws)
    return np.sum(np.power(Y-yHat, 2))


if __name__ == "__main__":
    # testMat = np.mat(np.eye(4))
    # mat0, mat1 = binSplitDataSet(testMat, 1, 0.5)
    # print(mat1)
    # myDat = loadDataSet('./data/ex2.txt ')
    # myMat = np.array(myDat)
    # myTree = createTree(myMat, ops=(0,1))
    # print(myTree)
    # myDatTest = loadDataSet('./data/ex2test.txt')
    # myDatTest = np.array(myDatTest)
    # myTree = prune(myTree, myDatTest)
    # print(myTree)

    myMat2 = np.array(loadDataSet('./data/exp2.txt'))
    print(createTree(myMat2, modelLeaf, modelErr, (1, 10)))