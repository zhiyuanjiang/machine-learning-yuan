import numpy as np
from SVM import svm

class optStruct:
    def __init__(self, dataMatIn, classLabels, C, toler):
        self.X = dataMatIn
        self.labelMat = classLabels
        self.C = C
        self.tol = toler
        self.m = np.shape(dataMatIn)[0]
        self.alphas = np.mat(np.zeros((self.m, 1)))
        self.b = 0
        # 保存误差Ek
        self.eCache = np.mat(np.zeros((self.m, 2)))

    def calcEk(self, k):
        fxk = float(np.multiply(self.alphas, self.labelMat).T * (self.X*self.X[k, :].T)) + self.b
        Ek = fxk - float(self.labelMat[k])
        return Ek

    def selectJ(self, i, Ei):
        maxK, maxDeltaE, Ej = -1, 0, 0
        self.eCache[i] = [1, Ei]
        validEcacheList = np.nonzero(self.eCache[:, 0].A)[0]
        if len(validEcacheList) > 1:
            # 在保存的误差eCache中寻找max|Ei-Ek|
            for k in validEcacheList:
                if k == i:
                    continue
                Ek = self.calcEk(k)
                deltaE = abs(Ei-Ek)
                if deltaE > maxDeltaE:
                    maxK = k
                    maxDeltaE = deltaE
                    Ej = Ek
            return maxK, Ej
        else:
            j = svm.selectJrand(i, self.m)
            Ej = self.calcEk(j)
        return j, Ej

    def updateEk(self, k):
        Ek = self.calcEk(k)
        self.eCache[k] = [1, Ek]

    # SMO的内层循环
    def innerL(self, i):
        Ei = self.calcEk(i)
        if ((self.labelMat[i] * Ei < -self.tol) and (self.alphas[i] < self.C)) or ((self.labelMat[i] * Ei > self.tol) and (self.alphas[i] > 0)):
            j, Ej = self.selectJ(i, Ei)
            # 这里应该不需要使用copy
            alphaIold = self.alphas[i].copy()
            alphaJold = self.alphas[j].copy()
            # 确定L和H
            if self.labelMat[i] != self.labelMat[j]:
                L = max(0, self.alphas[j] - self.alphas[i])
                H = min(self.C, self.C + self.alphas[j] - self.alphas[i])
            else:
                L = max(0, self.alphas[j] + self.alphas[i] - self.C)
                H = min(self.C, self.alphas[j] + self.alphas[i])
            if L == H:
                print("L==H")
                return 0
            eta = 2.0 * self.X[i, :] * self.X[j, :].T - self.X[i, :] * self.X[i, :].T - self.X[j,:] * self.X[j, :].T
            if eta >= 0:
                print("eta >= 0")
                return 0
            self.alphas[j] -= self.labelMat[j] * (Ei - Ej) / eta
            self.alphas[j] = svm.clipAlpha(self.alphas[j], H, L)
            self.updateEk(j)
            if abs(self.alphas[j] - alphaJold) < 0.00001:
                print("j not moving enough")
                return 0
            self.alphas[i] += self.labelMat[j] * self.labelMat[i] * (alphaJold - self.alphas[j])
            self.updateEk(i)
            b1 = self.b - Ei - self.labelMat[i] * (self.alphas[i] - alphaIold) * (self.X[i, :] * self.X[i, :].T) - self.labelMat[j] * (self.alphas[j] - alphaJold) * (self.X[i, :] * self.X[j, :].T)
            b2 = self.b - Ej - self.labelMat[i] * (self.alphas[i] - alphaIold) * (self.X[i, :] * self.X[i, :].T) - self.labelMat[j] * (self.alphas[j] - alphaJold) * (self.X[i, :] * self.X[j, :].T)
            if 0 < self.alphas[i] and self.C > self.alphas[i]:
                self.b = b1
            elif 0 < self.alphas[j] and self.C > self.alphas[j]:
                self.b = b2
            else:
                self.b = (b1 + b2) / 2.0
            return 1
        else:
            return 0

