import numpy as np

"""
loadDataSet:创建实验样本
:return
	postingList - list, 词条切分后的文档集合
	classVec - vector, 文档集合的分类标签
"""
def loadDataSet():
	postingList=[['my', 'dog', 'has', 'flea', 'problems', 'help', 'please'],
				['maybe', 'not', 'take', 'him', 'to', 'dog', 'park', 'stupid'],
				['my', 'dalmation', 'is', 'so', 'cute', 'I', 'love', 'him'],
				['stop', 'posting', 'stupid', 'worthless', 'garbage'],
				['mr', 'licks', 'ate', 'my', 'steak', 'how', 'to', 'stop', 'him'],
				['quit', 'buying', 'worthless', 'dog', 'food', 'stupid']]
	# 1代表侮辱性文字，0代表正常言论
	classVec = [0,1,0,1,0,1]
	return postingList,classVec


"""
createVocabList:创建一个词表

:param
	dataSet - list[list[str]], 词条切分后的文档集合
	
:return	
	- list, 词表
"""
def createVocabList(dataSet):
	vocabSet = set()
	for document in dataSet:
		vocabSet = vocabSet | set(document)
	return list(vocabSet)

"""
setOfWord2Vec:获得文档向量

:param:
	vocabList - list[str], 词汇表
	inputSet - list[str], 某个文档

:return 
	returnVec - list[int], 文档向量 
"""
def setOfWords2Vec(vocabList, inputSet):
	returnVec = [0]*len(vocabList)
	for word in inputSet:
		if word in inputSet:
			returnVec[vocabList.index(word)] = 1
		else:
			print('the word: {} is not in my vocablulary!'.format(word))
	return returnVec

"""
每个单词是否出现作为一个特征，这可以被描述为词集模型。
在词袋模型中，每个单词能出现多次。
setOfWords2vec:词集模型
bagOfWords2VecMN:词袋模型
"""
def bagOfWords2VecMN(vocabList, inputSet):
	returnVec = [0]*len(vocabList)
	for word in inputSet:
		if word in inputSet:
			returnVec[vocabList.index(word)] += 1
		else:
			print('the word: {} is not in my vocablulary!'.format(word))
	return returnVec

"""
trainNB0:计算P(wi|ci), P(ci)

:param 
	trainMatrix - list[list[int]], 训练用的文档向量
	trainCategory - list[int], 文档向量的类别
	
:return
	p0Vect - array, shape == (len(trainMatrix[0]),)), P(wi|c0), 单词wi在分类0的文档中的出现概率
	p1Vect - array, shape == (len(trainMatrix[0]),)), P(wi|c1)
	pAbusive - float, P(c1), 分类为1的文档在所有文档中出现的概率	
"""
def trainNB0(trainMatrix, trainCategory):
	numTrainDocs = len(trainMatrix)
	numWords = len(trainMatrix[0])
	# P(ci)的概率, ci = 1的概率
	pAbusive = sum(trainCategory)/float(numTrainDocs)
	# 分子初始化为1，分母初始化为2，避免出现P(wi|ci) = 0
	p0Num, p1Num = np.ones(numWords), np.ones(numWords)
	p0Denom, p1Denom = 2.0, 2.0
	for i in range(numTrainDocs):
		if trainCategory[i] == 1:
			p1Num += trainMatrix[i]
			p1Denom += np.sum(trainMatrix[i])
		else:
			p0Num += trainMatrix[i]
			p0Denom += np.sum(trainMatrix[i])
	# 计算的是P(wi|ci)
	# 进行对数处理，避免数据向下溢出
	p1Vect = np.log(p1Num/p1Denom)
	p0Vect = np.log(p0Num/p0Denom)
	return p0Vect, p1Vect, pAbusive

"""
classifyNB:对文档进行分类
	
:param
	vec2Classify - array, 某个文档向量
	p0Vec - P(wi|c0)
	p1Vec - P(wi|c1)
	pClass1 - P(c1)

:return
	- 分类结果
"""
def classifyNB(vec2Classify, p0Vec, p1Vec, pClass1):
	p1 = np.sum(vec2Classify * p1Vec) + np.log(pClass1)
	p0 = np.sum(vec2Classify * p0Vec) + np.log(1.0-pClass1)
	if p1 > p0:
		return 1
	else:
		return 0

def testingNB():
	listOPosts, listClasses = loadDataSet()
	myVocabList = createVocabList(listOPosts)
	trainMat = []
	for postinDoc in listOPosts:
		trainMat.append(setOfWords2Vec(myVocabList, postinDoc))
	p0V, p1V, pAb = trainNB0(np.array(trainMat), np.array(listClasses))
	testEntry = ['love', 'my', 'dalmation']
	thisDoc = np.array(setOfWords2Vec(myVocabList, testEntry))
	print(testEntry,'classified as: ', classifyNB(thisDoc, p0V, p1V, pAb))
	testEntry = ['stupid', 'garbage']
	thisDoc = np.array(setOfWords2Vec(myVocabList, testEntry))
	print(testEntry, 'classified as: ', classifyNB(thisDoc, p0V, p1V, pAb))

if __name__ == '__main__':
	listOposts, listClasses = loadDataSet()
	myVocabList = createVocabList(listOposts)
	# print(myVocabList)
	# v = setOfWords2Vec(myVocabList, listOposts[0])
	# print(v)
	# trainMat = []
	# for postinDoc in listOposts:
	# 	trainMat.append(setOfWords2Vec(myVocabList, postinDoc))
	# p0V, p1V, pAb = trainNB0(trainMat, listClasses)
	# print(pAb)
	# print(p0V)
	# print(p1V)
	testingNB()