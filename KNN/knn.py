import numpy as np
import operator
import os
import matplotlib.pyplot as plt

"""
KNN算法
(1) 计算已知类别数据集中的每个点与当前点之间的距离
(2) 按照距离递增次序排序
(3) 选取与当前点距离最小的k个点
(4) 确定前k个点所在类别的出现频率
(5) 返回前k个点出现频率最高的类别作为当前点的预测分类

"""

def createDataSet():
    group = np.array([[1.0, 1.1], [1.0, 1.0], [0, 0], [0, 0.1]])
    labels = ['A', 'A', 'B', 'B']
    return group, labels

def classify0(inX, dataSet, labels, k):
    dataSetSize = dataSet.shape[0]
    # np.tile(A, (a, b, c...)) 对数据A在a,b,c..轴上进行扩充
    diffMat = np.tile(inX, (dataSetSize, 1))-dataSet
    sqDiffMat = diffMat**2
    sqDistances = sqDiffMat.sum(axis=1)
    distances = sqDistances**0.5
    sortedDistIndicies = np.argsort(distances)
    classCount = {}
    for i in range(k):
        voteIlabel = labels[sortedDistIndicies[i]]
        classCount[voteIlabel] = classCount.get(voteIlabel, 0)+1
    # 对字典排序{'A':3, 'B':2, 'C':6} --> [('C',6), ('A',3),('B',2)]
    sortedClassCount = sorted(classCount.items(), key=operator.itemgetter(1), reverse=True)
    return sortedClassCount[0][0]


"""
函数说明：解析文件
didntLike  --> 1
smallDoses --> 2
largeDoses --> 3

params:
    filename:文件路径

return:
    returnMat:输入特征
    classLabelVector:分类标签
"""
def file2matrix(filename):
    with open(filename, 'r', encoding='utf-8') as fr:
        arrayOLines = fr.readlines()
        numberOfLines = len(arrayOLines)
        returnMat = np.zeros((numberOfLines, 3))
        classLabelVector = []
        index = 0
        for line in arrayOLines:
            line = line.strip()
            listFromLine = line.split('\t')
            # 读取的数据类型都是str，这里自动转换成了float
            returnMat[index, :] = listFromLine[0:3]
            # classLabelVector.append(int(listFromLine[-1]))
            if listFromLine[-1] == 'didntLike':
                classLabelVector.append(1)
            if listFromLine[-1] == 'smallDoses':
                classLabelVector.append(2)
            if listFromLine[-1] == 'largeDoses':
                classLabelVector.append(3)
            index += 1
        return returnMat, classLabelVector

"""
autoNorm: 对dataSet进行归一化

params:
    dataSet:特征集

return:
    normDataSet:标准化后的特征集
    ranges:最大值-最小值
    minVals:最小值
"""
def autoNorm(dataSet):
    # 在axis=0方向去最小值
    minVals = dataSet.min(0)
    maxVals = dataSet.max(0)
    ranges = maxVals-minVals
    normDataSet = np.zeros(np.shape(dataSet))
    # 样本数量
    m = dataSet.shape[0]
    # 对minVals进行扩充
    normDataSet = dataSet-np.tile(minVals, (m, 1))
    normDataSet = normDataSet/np.tile(ranges, (m, 1))
    return normDataSet, ranges, minVals

def datingClassTest(filename):
    # 测试数比例
    hoRatio = 0.10
    datingDataMat, datingLabels = file2matrix(filename)
    normMat, ranges, minVals = autoNorm(datingDataMat)
    m = normMat.shape[0]
    numTestVecs = int(m*hoRatio)
    errorCount = 0.0
    for i in range(numTestVecs):
        classifierResult = classify0(normMat[i, :], normMat[numTestVecs:m, :], datingLabels[numTestVecs:m], 3)
        print('the classifier came back with: %d, the real answer is: %d'%(classifierResult, datingLabels[i]))
        if(classifierResult != datingLabels[i]):
            errorCount += 1
    print('the total error rate is : %f'%(errorCount/float(numTestVecs)))


"""
img2vector:将一张图片转换成一个(1, 1024)的向量

params:
    filename:文件路径

return 
    returnVect:转换后的向量
"""

def img2vector(filename):
    returnVect = np.zeros((1, 1024))
    with open(filename) as fr:
        for i in range(32):
            lineStr = fr.readline()
            for j in range(32):
                returnVect[0, 23*i+j] = int(lineStr[j])
        return returnVect

"""
handwritingClassTest:测试手写数字的正确率
"""
def handwritingClassTest():
    root = 'f:\\machine-learning\\data\\KNN\\'
    hwLabels = []
    trainingFileList = os.listdir(root+'trainingDigits')
    m = len(trainingFileList)
    trainingMat = np.zeros((m, 1024))
    for i in range(m):
        fileNameStr = trainingFileList[i]
        fileStr = fileNameStr.split('.')[0]
        classNumStr = int(fileStr.split('_')[0])
        hwLabels.append(classNumStr)
        trainingMat[i,:] = img2vector(root+'trainingDigits\\'+fileNameStr)

    testFileList = os.listdir(root+'testDigits')
    errorCount = 0.0
    mTest = len(testFileList)
    for i in range(mTest):
        fileNameStr = testFileList[i]
        fileStr = fileNameStr.split('.')[0]
        classNumStr = int(fileStr.split('_')[0])
        vectorUnderTest = img2vector(root+'testDigits\\'+fileNameStr)
        classifierResult = classify0(vectorUnderTest, trainingMat, hwLabels, 3)
        print('the classifier came back with: %d, the real answer is: %d'%(classifierResult, classNumStr))
        if classifierResult != classNumStr:
            errorCount += 1
    print('\nthe total number of errors is: %d'%(errorCount))
    print('\nthe total error rate is: %f'%(errorCount/float(mTest)))

if __name__ == '__main__':
    filename = 'f:\\machine-learning\\data\\KNN\\datingTestSet.txt'
    datingDataMat, datingLabels = file2matrix(filename)
    # 可视化
    # s:shape, c:color
    # plt.scatter(datingDataMat[:,0], datingDataMat[:,1], s=15.0*np.array(classLabelVector), c=15.0*np.array(classLabelVector))
    # plt.show()
    # 归一化
    normMat, ranges, minVals = autoNorm(datingDataMat)
    # print(normMat)
    # print(ranges)
    # print(minVals)

    # datingClassTest(filename)

    handwritingClassTest()