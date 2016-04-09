# -*- coding: utf-8 -*-
"""
Created on Fri Apr  8 23:59:45 2016

@author: wq

逻辑回归
"""
from numpy import  *


class LogisticRegression():
    '''
    二项逻辑回归模型
    
    参数
    --------
    
    '''
    def __init__(self, alpha=1):
        '''
        初始化
        参数
        ---------
        alpha,梯度下降迭代步长
        '''
        self.alpha = alpha
        self.weights = []     #模型权值
        
    def sigmoid(self, inX):
        return 1/(1+sum(exp(-inX)))
   
    def fit(self, dataMatrix, classLabels):
        '''根据训练数据拟合模型
        
        参数
        ------------
        dataMatrix:
            训练样本属性矩阵
            类型：array
        classLabels：
            训练样本标签
            类型：list

        '''
        
        row,col = dataMatrix.shape
        if len(self.weights) == 0:
            weights = ones(col)   #初始化权值矩阵2
        else:
            weights = self.weights
        diff = weights
        j = 0 #迭代次数
        while(abs(max(diff))>0.01):
            dataIndex = range(row)
            for i in range(row):
                alpha = 4/(self.alpha+j+i)+0.0001    #alpha随着每次迭代，下降。参数趋于稳定
                randIndex = int(random.uniform(0,len(dataIndex)))#随机选取训练对象
                h = self.sigmoid(sum(dataMatrix[randIndex]*weights))#矩阵
                error = classLabels[randIndex] - h
                
                diff = weights
                weights = weights + alpha * error * dataMatrix[randIndex]
                diff = weights - diff                
                j += 1
                del(dataIndex[randIndex])
        self.weights = weights
        #print weights
        
    def predict(self,dataList):
        '''
        训练的二项逻辑回归预测
        
        参数
        --------
        dataList：
            待预测样本属性,
            类型：list
        '''
        if len(self.weights) == 0:
            raise KeyError,("没有进行模型训练")
        prob = self.sigmoid(sum(dataList*self.weights))
        if prob > 0.5:return 1
        else: return 0
        
    def score(self,dataMatrix, classLabels):
        '''
        评价模型性能
        
        参数
        ------------
        dataMatrix:
            测试样本属性举证
            属性：array
        classLabels:
            测试样本类别，这里只有1、0两类
            属性：list
        '''
        numData = len(dataMatrix)
        errorCount = 0
        j = 0 #行数记录，
        for line in dataMatrix:
            #print '预测结果',self.predict(line)
            #print '样本标签',classLabels[j]
            if int(self.predict(line)) != int(classLabels[j]):
                errorCount += 1
            j += 1
        #print errorCount
        errorRate = (float(errorCount)/numData)
        print "the error rate of this test is: %f" % errorRate
        return errorRate
        
if __name__=='__main__':
    frTrain = open('horseColicTraining.txt'); 
    frTest = open('horseColicTest.txt')
    trainingSet = []; trainingLabels = []
    for line in frTrain.readlines():
        line = line.strip().split('\t')
        trainingSet.append(map(float, line)[:-1])
        trainingLabels.append(float(line[21]))
    
    testSet = [];testLabels = []
    for line in frTest:
        line = line.strip().split('\t')
        testSet.append(map(float, line)[:-1])
        testLabels.append(line[21])

    clf = LogisticRegression(1.0)
    for i in range(10):
        clf.fit(array(trainingSet), trainingLabels)
        clf.score(array(testSet), testLabels)
