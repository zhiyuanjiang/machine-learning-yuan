import matplotlib.pyplot as plt
from ID3 import id3

decisionNode = dict(boxstyle='sawtooth', fc='0.8')
leafNode = dict(boxstyle='round4', fc='0.8')
arrow_args = dict(arrowstyle='<-')


"""
plotNode:

params:
    nodeTxt:节点信息
    centerPt:终点坐标
    parenter:起点坐标
    nodeType:节点形状
"""
def plotNode(nodeTxt, centerPt, parentPt, nodeType):
    createPlot.ax1.annotate(nodeTxt, xy=parentPt, xycoords='axes fraction', xytext=centerPt,
                            textcoords='axes fraction', va='center', ha='center', bbox=nodeType, arrowprops=arrow_args)

# def createPlot():
#     fig = plt.figure(1, facecolor='white')
#     fig.clf()
#     createPlot.ax1 = plt.subplot(111, frameon=False)
#     plotNode('decision node', (0.5, 0.1), (0.1, 0.5), decisionNode)
#     plotNode('leaf node', (0.8, 0.1), (0.3, 0.8), leafNode)
#     plt.show()

def retrieveTree(i):
    listOfTrees = [{'no surfacing':{0:'no', 1:{'flippers':{0:'no', 1:'yes'}}}},
                   {'no surfacing':{0:'no', 1:{'flippers':{0:{'head':{0:'no', 1:'yes'}}, 1:'no'}}}}]
    return listOfTrees[i]

"""
getNumLeafs:获取一棵树的叶子节点数目(使用深度搜索)

params:
    myTree:使用嵌套字典构建的树
    
return:
    numLeafs:叶子节点的数目
"""
def getNumLeafs(myTree):
    numLeafs = 0
    firstStr = list(myTree.keys())[0]
    secondDict = myTree[firstStr]
    for key in secondDict.keys():
        if type(secondDict[key]).__name__  == 'dict':
            numLeafs += getNumLeafs(secondDict[key])
        else:
            numLeafs += 1
    return numLeafs

"""
getTreeDepth:获取一棵树的深度

param:
    myTree:使用嵌套字典构建的树
    
return:
    maxDepth:树中最大的深度
"""
def getTreeDepth(myTree):
    maxDepth = 0
    firstStr = list(myTree.keys())[0]
    secondDict = myTree[firstStr]
    for key in secondDict.keys():
        if type(secondDict[key]).__name__ == 'dict':
            thisDepth = 1+getTreeDepth(secondDict[key])
        else:
            thisDepth = 1
        # 整个函数是一个代码块，会对整个函数一起解释，所以thisDepth会被识别到(根c++不一样)
        if thisDepth > maxDepth:
            maxDepth = thisDepth
    return maxDepth


def plotMidText(cntrPt, parentPt, txtString):
    xMid = (parentPt[0]+cntrPt[0])/2.0
    yMid = (parentPt[1]+cntrPt[1])/2.0
    createPlot.ax1.text(xMid, yMid, txtString)

def plotTree(myTree, parentPt, nodeTxt):
    numLeafs = getNumLeafs(myTree)
    depth = getTreeDepth(myTree)
    firstStr = list(myTree.keys())[0]
    # 这里的cntrPt为什么是这样计算的?
    cntrPt = (plotTree.xOff+(1.0+float(numLeafs))/2.0/plotTree.totalW, plotTree.yOff)
    plotMidText(cntrPt, parentPt, nodeTxt)
    plotNode(firstStr, cntrPt, parentPt, decisionNode)
    sencondDict = myTree[firstStr]
    # 决策树每一层高度递减
    plotTree.yOff = plotTree.yOff-1.0/plotTree.totalD
    for key in sencondDict.keys():
        if type(sencondDict[key]).__name__ == 'dict':
            plotTree(sencondDict[key], cntrPt, str(key))
        else:
            plotTree.xOff = plotTree.xOff+1.0/plotTree.totalW
            plotNode(sencondDict[key], (plotTree.xOff, plotTree.yOff), cntrPt, leafNode)
            plotMidText((plotTree.xOff, plotTree.yOff), cntrPt, str(key))
    plotTree.yOff = plotTree.yOff+1.0/plotTree.totalD


def createPlot(inTree):
    fig = plt.figure(1, facecolor='white')
    fig.clf()
    axprops = dict(xticks=[], yticks=[])
    createPlot.ax1 = plt.subplot(111, frameon=False, **axprops)
    plotTree.totalW = float(getNumLeafs(inTree))
    plotTree.totalD = float(getTreeDepth(inTree))
    plotTree.xOff = -0.5/plotTree.totalW
    plotTree.yOff = 1.0
    plotTree(inTree, (0.5,  1.0), ' ')
    plt.show()


"""
classify:分类器

:param
    inputTree - 决策树
    featLabels - 标签列表
    testVec - 测试数据

:return
    classLabel - 分类结果
"""
def classify(inputTree, featLabels, testVec):
    firstStr = list(inputTree.keys())[0]
    secondDict = inputTree[firstStr]
    featIndex = featLabels.index(firstStr)
    for key in secondDict.keys():
        if testVec[featIndex] == key:
            if type(secondDict[key]).__name__ == 'dict':
                classLabel = classify(secondDict[key], featLabels, testVec)
            else:
                classLabel = secondDict[key]
    return classLabel

"""
storeTree:存储决策树

:param
    inputTree - 决策树
    filename - 存储路径
    
"""
def storeTree(inputTree, filename):
    import pickle
    with open(filename, 'wb') as fw:
        pickle.dump(inputTree, fw)
        fw.close()

"""
grabTree:读取决策树

:param
    filename - 文件路径
    
:return
    嵌套字典形式的决策树
"""
def grabTree(filename):
    import pickle
    with open(filename, 'rb') as fr:
        return pickle.load(fr)

if __name__ == '__main__':
    # createPlot()
    # myTree = retrieveTree(0)
    # numLeafs = getNumLeafs(myTree)
    # print(numLeafs)
    # depth = getTreeDepth(myTree)
    # print(depth)
    # createPlot(myTree)
    # myDat, labels = id3.createDataSet()
    # myTree = retrieveTree(0)
    # ans = classify(myTree, labels, [1, 0])
    # print(ans)

    # myTree = retrieveTree(0)
    # storeTree(myTree, 'classifierStorage.txt')
    # print(grabTree('classifierStorage.txt'))

    with open('f:\\machine-learning\\data\\ID3\\lenses.txt') as f:
        lenses = [inst.strip().split('\t') for inst in f.readlines()]
        lensesLabels = ['age', 'prescript', 'astigmatic', 'tearRate']
        lensesTree = id3.createTree(lenses, lensesLabels)
        print(lensesTree)
        createPlot(lensesTree)