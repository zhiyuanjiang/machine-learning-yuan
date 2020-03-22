from BAYES import bayes
import random
import numpy as np

"""
textParse:对一个文本进行转换

:return
    - list[str]
"""
def textParse(bigString):
    import re
    listOfTokens = re.split(r'\W+', bigString)
    return [tok.lower() for tok in listOfTokens if len(tok) > 2]

"""
spamTest:构建一个垃圾邮件识别的分类器，并测试正确率
"""
def spamTest():
    docList, classList, fullText = [], [], []
    for i in range(1, 26):
        wordList = textParse(open('F:\\machine-learning\\data\\BAYES\\email\\spam\\%d.txt'%(i)).read())
        docList.append(wordList)
        fullText.extend(wordList)
        classList.append(1)
        wordList = textParse(open('F:\\machine-learning\\data\\BAYES\\email\\ham\\%d.txt'%(i)).read())
        docList.append(wordList)
        fullText.extend(wordList)
        classList.append(0)
    vocabList = bayes.createVocabList(docList)
    trainingSet, testSet = list(range(50)), []
    # 随机采样
    for i in range(10):
        randIndex = int(random.uniform(0, len(trainingSet)))
        testSet.append(trainingSet[randIndex])
        del(trainingSet[randIndex])
    trainMat, trainClasses = [], []
    for docIndex in trainingSet:
        trainMat.append(bayes.setOfWords2Vec(vocabList, docList[docIndex]))
        trainClasses.append(classList[docIndex])
    p0V, p1V, pSpam = bayes.trainNB0(np.array(trainMat), np.array(trainClasses))
    errorCount = 0
    for docIndex in testSet:
        wordVector = bayes.setOfWords2Vec(vocabList, docList[docIndex])
        if bayes.classifyNB(np.array(wordVector), p0V, p1V, pSpam) != classList[docIndex]:
            errorCount += 1
            print(docList[docIndex])
    print('the error rate is: ', float(errorCount)/len(testSet))


if __name__ == '__main__':
    spamTest()
