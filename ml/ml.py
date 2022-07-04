import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import random


def calcEk(oS, k):
    g_x = np.multiply(oS.alphas, oS.dataLabels).T*oS.kernel[:, k] + oS.b
    e_k = g_x - oS.dataLabels[k]
    return e_k  


def selectJ(i, oS, Ei):
    max_k = -1
    max_deltaE = 0
    Ej = 0
    oS.eCache[i] = [0, Ei]
    validEcacheList = np.nonzero(oS.eCache[:, 0].A)[0]
    if len(validEcacheList) > 1:
        for k in validEcacheList:
            if k == i:
                continue
            Ek = calcEk(oS, k)
            deltaE = abs(Ek-Ei)
            if deltaE > max_deltaE:
                max_k = k
                max_deltaE = deltaE
                Ej = Ek
        return max_k, Ej
    else:
        j = i
        while j == i:
            j = int(random.uniform(0, oS.m))
        Ej = calcEk(oS, j)
        return j, Ej


def updateEk(oS, j):
    Ek = calcEk(oS, j)
    oS.eCache[j] = [1, Ek]
    return


def innerL(i, oS):
    Ei = calcEk(oS, i)
    g_x = np.multiply(oS.alphas, oS.dataLabels).T*oS.kernel[:, i] + oS.b
    # 使用SMO算法求解alpha
    if not ((oS.dataLabels[i]*g_x >= 1 and oS.alphas[i] == 0) or (oS.dataLabels[i]*g_x == 1 and 0 < oS.alphas[i] < oS.c) or (oS.dataLabels[i]*g_x <= 1 and oS.alphas[i] == oS.c)): # KKT条件
        # 选择第二参数j的时候，希望Ej-Ei越大越好
        j, Ej = selectJ(i, oS, Ei)
        alphaIold = oS.alphas[i].copy()
        alphaJold = oS.alphas[j].copy()
        if oS.dataLabels[i] != oS.dataLabels[j]:
            L = max(0, oS.alphas[j] - oS.alphas[i])
            H = min(oS.c, c+oS.alphas[j] - oS.alphas[i])
        else:
            L = max(0, oS.alphas[j] + oS.alphas[i] - oS.c)
            H = min(oS.c, oS.alphas[j] + oS.alphas[i])
        if L == H:
            print("L==H")
            return 0
        eta =  oS.kernel[i, i] + oS.kernel[j, j] - 2 * oS.kernel[i, j]
        alphaJnew = alphaJold + oS.dataLabels[j] * (Ei-Ej) / eta
        if alphaJnew > H:
            alphaJnew = H
        elif alphaJnew < L:
            alphaJnew = L
        else:
            alphaJnew = alphaJnew
        oS.alphas[j] = alphaJnew
        updateEk(oS, j)
        if abs(alphaJnew - alphaIold) < oS.threshold:
            print("j not moving enough")
            return 0
        alphaInew = alphaIold + oS.dataLabels[i] * oS.dataLabels[j] * (alphaJold - alphaJnew)
        oS.alphas[i] = alphaInew
        updateEk(oS, i)
        # 求解b
        b1 = oS.b - Ei - oS.dataLabels[i]*(alphaInew - alphaIold)*oS.kernel[i, i] - oS.dataLabels[j]*(alphaJnew - alphaJold)*oS.kernel[j, j]
        b2 = oS.b - Ej - oS.dataLabels[i]*(alphaInew - alphaIold)*oS.kernel[i, j] - oS.dataLabels[j]*(alphaJnew - alphaJold) * oS.kernel[j, j]
        if 0 < alphaInew < oS.c:
            oS.b = b1
        elif 0 < alphaJnew < oS.c:
            oS.b = b2
        else:
            oS.b = (b1+b2) / 2
        # 更新模型的支持向量及标签
        svInd = np.nonzero(oS.alphas)[0]
        oS.svs = oS.dataMat[svInd]
        oS.labelSV = oS.dataLabels[svInd]
        oS.svInd = svInd
        print("thre are %d support vectors" % oS.svs.shape[0])
        return 1
    else:
        return 0


def predict(oS, test_data_mat, test_label_mat):
    m_test = test_data_mat.shape[0]
    errot_count = 0
    for i in range(m_test):
        tem = np.mat(np.zeros(shape=(oS.svs.shape[0], 1)))
        x_i = test_data_mat[i, :]
        for j in range(oS.svs.shape[0]):
            delta = np.mat(oS.svs[j, :] - x_i)
            tem[j] = delta*delta.T
        tem = np.exp(tem/(-1*oS.sigma**2))
        # test_kernel[:, i] = tem
        predict_val = tem.T * np.multiply(oS.labelSV, oS.alphas[oS.svInd]) + oS.b
        if np.sign(predict_val) != np.sign(test_label_mat[i]):
            errot_count += 1
    print("the test error rate is : ", errot_count)
    return errot_count


class optStruct(object):
    def __init__(self, dataMat, dataLabels, c, threshold):
        self.dataMat = dataMat
        self.dataLabels = dataLabels
        self.sigma = 1.6
        self.c = c
        self.threshold = threshold
        self.row = dataMat.shape[0]
        self.alphas = np.mat(np.zeros(shape=(self.row, 1)))
        self.b = 0
        self.eCache = np.mat(np.zeros(shape=(self.row, 2)))
        self.kernel = np.mat(np.zeros(shape=(self.row, self.row)))
        self.svs = []
        self.labelSV = []
        self.svInd = []
        for i in range(self.row):
            self.kernel[:, i] = kernelTrans(self.dataMat, self.sigma, self.dataMat[i, :])


def kernelTrans(dataMat, sigma, x_i):
    m = dataMat.shape[0]
    tem = np.mat(np.zeros(shape=(dataMat.shape[0], 1)))
    for j in range(m):
        delta = np.mat(dataMat[j, :] - x_i)
        tem[j] = delta*delta.T
    tem = np.exp(tem/(-1*sigma**2))
    return tem


def svm_train(dataMat, labelMat, c, threshold):
    oS = optStruct(dataMat, labelMat, c, threshold)
    while (iter < 50 and alpha_pair_changed > 0) or entireset:
        alpha_pair_changed = 0
        if entireset:
            for i in range(oS.m):
                alpha_pair_changed += innerL(i, oS)
            entireset = False
        else:
            nonBoundIs = np.nonzero((oS.alphas.A > 0) * (oS.alphas.A < c))[0]
            for i in nonBoundIs:
                alpha_pair_changed += innerL(i, oS)
        iter += 1
    return oS


def svm_predict(oS, test_data_mat, test_label_mat):
    return predict(oS, test_data_mat, test_label_mat)