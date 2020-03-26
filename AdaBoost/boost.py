import numpy as np
from AdaBoost import adaboost


def stumpClassify(dataMatrix, dimen, threshVal, threshIneq):
    """
    对样本进行简单的划分，如果样本的dimen维特征的value >= or <= 某个边界值，则预测为负样本，否则为正样本
    :param dataMatrix: ndarray (m, n), 样本数据
    :param dimen: int, 选择的哪一维特征
    :param threshVal: float, 划分样本的临界值
    :param threshIneq: str, 划分方式
    :return:
        retArray - ndarray (m, 1), 对样本的分类结果，1:正样本，-1:负样本
    """
    retArray = np.ones((dataMatrix.shape[0], 1))
    if threshIneq == 'lt':
        retArray[dataMatrix[:, dimen] <= threshVal] = -1.
    else:
        retArray[dataMatrix[:, dimen] >  threshVal] = -1.
    return retArray

def buildStrump(dataArr, classLabels, D):
    """
    构建最佳的单层决策树
    :param dataArr: list[list], 样本数据
    :param classLabels: list, 标签
    :param D: 每个样本的权重向量
    :return:
        bestStump - dict, 记录最佳单层决策树信息
        minError - 最小的权重误差
        bestClassEst - 最佳的样本分类结果
    """
    dataMatrix = np.mat(dataArr)
    labelMat = np.mat(classLabels).T
    m, n = dataMatrix.shape
    numSteps = 10.0
    bestStump = {}
    bestClassEst = np.mat(np.zeros((m, 1)))
    minError = np.inf
    # 检测n个特征
    for i in range(n):
        rangeMin = dataMatrix[:, i].min()
        rangeMax = dataMatrix[:, i].max()
        stepSize = (rangeMax-rangeMin) / numSteps
        for j in range(-1, int(numSteps)+1):
            for inequal in ['lt', 'gt']:
                threshVal = (rangeMin+float(j)*stepSize)
                # 对样本进行划分
                predictedVals = stumpClassify(dataMatrix, i, threshVal, inequal)
                errArr = np.mat(np.ones((m, 1)))
                # 标记预测错误的样本为0
                errArr[predictedVals == labelMat] = 0
                # 计算带权重的错误率，增加之前分类器分类错误的样本权重，这样就会尽可能选择对该样本分类正确的分类器。
                weightedError = D.T*errArr
                # print("split: dim %d, thresh %.2f, thresh ineqal: %s, the weighted error is %.3f"%(i, threshVal, inequal, weightedError))
                if weightedError < minError:
                    minError = weightedError
                    bestClassEst = predictedVals.copy()
                    bestStump['dim'] = i
                    bestStump['thresh'] = threshVal
                    bestStump['ineq'] = inequal
    return bestStump, minError, bestClassEst


if __name__ == "__main__":
    D = np.mat(np.ones((5, 1)))/5.
    datMat, classLabels = adaboost.loadSimpData()
    buildStrump(datMat, classLabels, D)