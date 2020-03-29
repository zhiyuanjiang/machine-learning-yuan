import matplotlib.pyplot as plt
import numpy as np
from AdaBoost import boost

root = 'F:\\machine-learning\\data\\AdaBoost\\'

def loadSimpData():
    """
    生成简单的数据集
    :return:
        datMat - array shape=(5,2)
        classLabels - list[float], 标签
    """
    datMat = np.array([[1., 2.1],
                       [2., 1.1],
                       [1.3, 1.],
                       [1. , 1.],
                       [2., 1. ]])
    classLabels = [1., 1., -1., -1., 1.]
    return datMat, classLabels


def loadDataSet(filename):
    """
    加载数据
    :param filename:
    :return:
    """
    dataMat, labelMat = [], []
    with open(filename, 'r') as fr:
        for line in fr.readlines():
            lineArr = []
            curLine = line.strip().split('\t')
            for i in range(len(curLine)-1):
                lineArr.append(float(curLine[i]))
            dataMat.append(lineArr)
            labelMat.append(float(curLine[-1]))
        return dataMat, labelMat

def adaBoostTrainDS(dataArr, classLabels, numIt=40):
    """
    adaBoost模型训练
    :param dataArr: list[list], 样本数据
    :param classLabels: list, 标签
    :param numIt: 迭代次数
    :return:
        weekClassArr - list[dict], 使用字典表示的多个单层决策树分类器
    """
    weakClassArr = []
    m = len(classLabels)
    # D是一个概率分布，保证np.sum(D) = 0
    D = np.mat(np.ones((m, 1))/m)
    aggClassEst = np.mat(np.zeros((m, 1)))
    for i in range(numIt):
        bestStump, error, classEst = boost.buildStrump(dataArr, classLabels, D)
        # print("D:", D.T)
        # max(error, 1e-6) 防止除0错误
        alpha = float(0.5*np.log((1.-error)/max(error, 1e-6)))
        bestStump['alpha'] = alpha
        # 添加生成的单层决策树弱分类器
        weakClassArr.append(bestStump)
        # print("classEst: ", classEst.T)
        expon = np.multiply(-1.*alpha*np.mat(classLabels).T, classEst)
        D = np.multiply(D, np.exp(expon))
        D = D/D.sum()
        aggClassEst += alpha*classEst
        aggErrors = np.multiply(np.sign(aggClassEst) != np.mat(classLabels).T, np.ones((m, 1)))
        errorRate = aggErrors.sum()/m
        print("total error: ", errorRate, "\n")
        if errorRate == 0.:
            break
    print(weakClassArr)
    return weakClassArr, aggClassEst


def adaClassify(dataToClass, classifierArr):
    """
    对测试样本进行分类
    :param dataToClass: test data
    :param classifierArr: 弱分类器组
    :return:
        - 分类结果
    """
    dataMatrix = np.mat(dataToClass)
    m = dataMatrix.shape[0]
    aggClassEst = np.mat(np.zeros((m, 1)))
    for i in range(len(classifierArr)):
        classEst = boost.stumpClassify(dataMatrix, classifierArr[i]['dim'], classifierArr[i]['thresh'], classifierArr[i]['ineq'])
        aggClassEst += classifierArr[i]['alpha']*classEst
        # print(aggClassEst)
    return np.sign(aggClassEst)

def myplot(datMat, classLabels):
    y = [(t+2)*20 for t in classLabels]
    plt.scatter(datMat[:,0], datMat[:,1], c=y)
    plt.show()


def plotROC(predStrengths, classLabels):
    """
    绘制ROC曲线，计算AUC值
    ROC曲线：先从右上角画起，假如所有的样本预测都为真，此时的假阳率和真阳率都为1，第一个点为（1,1）
    对样本预测大小进行排序，每次取一个最小的样本预测为假，如果样本为假，那么此时假阳率-1/反例数；如果样本为真，此时真阳率-1/正例数
    AUC：ROC曲线下的面积
    :param predStrengths: ndarray shape=(n, ) 样本预测值
    :param classLabels: ndarray shape=(n, ), 标签
    :return:
    """
    import matplotlib.pyplot as plt
    cur = (1., 1.)
    ySum = 0.
    # classLabels=1 为正样本
    numPosClas = np.sum(classLabels == 1.)
    yStep = 1./float(numPosClas)
    xStep = 1./float(len(classLabels)-numPosClas)
    # 从小到大排序
    sortedIndicies = predStrengths.argsort()
    fig = plt.figure()
    fig.clf()
    ax = plt.subplot(111)
    for index in sortedIndicies.tolist()[0]:
        if classLabels[index] == 1.:
            delX = 0;
            delY = yStep
        else:
            delX = xStep
            delY = 0
            # 计算所有的真阳率的和
            ySum += cur[1]
        ax.plot([cur[0], cur[0]-delX], [cur[1], cur[1]-delY], c='b')
        cur = (cur[0]-delX, cur[1]-delY)
    ax.plot([0, 1], [0, 1], 'b--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC curve for AdaBoost Horse Colic Detection System')
    ax.axis([0, 1, 0, 1])
    plt.show()
    print("the Area Under the Curve is: ", ySum*xStep)

def plotROC2(predStrengths, classLabels):
    """
    使用sklearn中的roc_curve, auc
    :param predStrengths: ndarray shape=(n, ), 预测值
    :param classLabels: ndarray shape=(n, ), 标签
    :return:
    """
    from sklearn.metrics import roc_curve, auc  ###计算roc和auc
    fpr, tpr, threshold = roc_curve(classLabels, predStrengths, pos_label=1)  ###计算真正率和假正率
    roc_auc = auc(fpr, tpr)  ###计算auc的值

    plt.figure()
    lw = 2
    plt.figure(figsize=(10, 10))
    plt.plot(fpr, tpr, color='darkorange',
             lw=lw, label='ROC curve (area = %0.2f)' % roc_auc)  ###假正率为横坐标，真正率为纵坐标做曲线
    plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver operating characteristic example')
    plt.legend(loc="lower right")
    plt.show()

    


if __name__ == "__main__":
    # dataArr, labelArr = loadSimpData()
    # myplot(dataArr, labelArr)
    # classifierArr = adaBoostTrainDS(dataArr, labelArr, 9)
    # v = adaClassify([0, 0], classifierArr)
    # print(v)
    datArr, labelArr = loadDataSet(root+'horseColicTraining2.txt')
    classifierArray, aggClassEst = adaBoostTrainDS(datArr, labelArr, 50)
    # testArr, testLabelArr = loadDataSet(root+'horseColicTest2.txt')
    # prediction10 = adaClassify(testArr, classifierArray)
    # m = prediction10.shape[0]
    # errArr = np.mat(np.ones((m, 1)))
    # assert prediction10.shape == (m, 1)
    # testLabelArr = np.mat(testLabelArr).T
    # assert testLabelArr.shape == (m, 1)
    # errorRate = errArr[prediction10 != testLabelArr].sum()/m
    # print(errorRate)
    plotROC(aggClassEst.T, np.array(labelArr))
    # plotROC2(np.squeeze(aggClassEst.A), np.array(labelArr))
