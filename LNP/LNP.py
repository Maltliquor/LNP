#!/usr/bin/env python
# vim: set fileencoding=utf-8 :

import random
import numpy as np
from cvxopt import matrix, solvers
import math

class LNP:

    def __init__(self):
        self.X = []
        self.Y = []               # 每一个样本的标签
        self.attr_num = 8             # 样本属性的数量
        self.label_num = 2        # 标签的数量
        self.sigma = 5            # 距离的sigma值
        self.alpha = 0.95
        self.maxk = 3             # 最近邻的邻居数
        self.sub = []             # 每一个属性最小值与最大值的差，用于归一化
        self.gram = []            # gram矩阵（G矩阵）
        self.W = []               # 每个样本的权重矩阵
        self.percentage_label_sample = 0.1 # 已有确定标签的样本数量的百分比
        self.num_label_sample = 0 # 已有确定标签的样本数量
        self.num_samples = 0

        self.neighbor = []        # 记录每个点的k个最近邻
        self.neighbor_num = 3     # 每个样本的近邻数量
        self.num = 0

    def set_filename_and_op(self, filename, separator):
        self.filename = filename        # 记录样本的文件名
        self.separator = separator      # 记录样本数据的分割符

    def read_data(self):
        with open(self.filename) as f:
            lines = f.readlines()       # 读取所有样本数据

            for line in lines:
                if self.separator != '\t':
                    line = line.replace(self.separator, '\t')
                content = line.split('\t')   # 根据分割符分割每个属性的数据
                new_item = []
                for i in range(self.attr_num):
                    num = float(content[i])
                    new_item.append(num)     # 往item中放入每一个属性的值
                label = content[self.attr_num]
                label = label.replace('\n', '')
                labeli = int(label)
                if labeli == 0:
                    labeli = -1
                self.Y.append(labeli)      # 将标签赋予对应的样本
                self.X.append(new_item)    # 往矩阵里存入样本
            print(self.Y)
            self.num_samples = len(self.X)

    def max_min(self):
        num_samples = len(self.X)
        temp1 = []
        # 找出每个属性最大值与最小值的差，用于归一化
        for i in range(self.attr_num):
            for j in range(num_samples):
                temp1.append(self.X[j][i])
            max_temp = max(temp1)
            min_temp = min(temp1)
            self.sub.append(max_temp-min_temp)
            temp1 = []

    # 返回X矩阵
    def get_x(self):
        return self.X

    # 返回Y矩阵
    def get_y(self):
        return self.Y

    def build_graph(self):
        self.max_min()
        num_samples = len(self.X)

        self.affinity_matrix =[[0 for col in range(self.num_samples)] for row in range(self.num_samples)]

        for i in range(num_samples):
            self.affinity_matrix[i][i] = [0.0, i]
            for j in range(num_samples):
                diff = 0.0
                for k in range(8):
                    if i != j:
                        dist = self.X[i][k] - self.X[j][k]
                        dist = dist/self.sub[k]
                        diff += dist ** 2

                # self.gram[i][j] = diff       # 同样的j不同的k,gram值是一样的,注意这里diff已经归一化，可以考虑不归一化
                if i != j:
                    self.affinity_matrix[i][j] = [math.exp(diff/ (-2.0 * (self.sigma ** 2))), j]


    def set_neighbor(self):
        num_samples = len(self.X)

        self.neighbor = [[]for row in range(num_samples)]
        for i in range(num_samples):
            temp = sorted(self.affinity_matrix[i], key=lambda x: x[0])
            temp.reverse()
            for k in range(self.maxk):
                j = temp[k][1]
                self.neighbor[i].append(j)
            if i == 1:
                print(self.X[i])
                print(self.X[self.neighbor[i][0]])
        print(self.neighbor)

    def set_gram(self):
        self.gram = [[0 for col in range(self.maxk)] for row in range(self.num_samples)]
        for i in range(self.num_samples):
            for j in range(self.maxk):
                neighbor = self.neighbor[i][j]
                diff = 0.0
                for k in range(8):
                    dist = self.X[i][k] - self.X[neighbor][k]
                    dist = dist / self.sub[k]
                    diff += dist ** 2
                self.gram[i][j] = diff  # 同样的j不同的k,gram值是一样的
        print(self.gram)

    def solve_weight(self):
        self.set_gram()
        self.W = np.zeros((self.num_samples, self.num_samples), np.float32)
        for i in range(self.num_samples):
            self.cal_weight(i)

    def cal_weight(self, i):
        # 构造二次规划的Q矩阵

        '''
        tempQ = np.zeros((self.maxk, self.maxk), np.double)
        for j in range(self.maxk):
            tempQ[j][j] = self.gram[i][j]
        for m in range(self.maxk):
            for n in range(self.maxk):
                tempQ[m][n] = (self.gram[i][m] + self.gram[i][n]) / 2
        print(type(tempQ))
        Q = 2 * matrix(tempQ)
        tempp = np.zeros((1, self.maxk), np.double)
        p = matrix(tempp)                        # 代表一次项的系数
        print(p)
        G = matrix([[-1.0, 0.0, 0.0], [0.0, -1.0, 0.0], [0.0, 0.0, -1.0]])        # G和h代表GX+s = h，s>=0,表示每一个变量x均大于零
        temph = np.zeros((1, self.maxk), np.double)
        h = matrix(temph)
        tempA = np.ones((1, self.maxk), np.double)
        A = matrix(tempA)

        '''

        q11 = self.gram[i][0]  # 对角线为二次方的系数
        q12 = (self.gram[i][0] + self.gram[i][1]) / 2
        q13 = (self.gram[i][0] + self.gram[i][2]) / 2
        q21 = (self.gram[i][0] + self.gram[i][1]) / 2
        q22 = self.gram[i][1]
        q23 = (self.gram[i][1] + self.gram[i][2]) / 2
        q31 = (self.gram[i][0] + self.gram[i][2]) / 2
        q32 = (self.gram[i][1] + self.gram[i][2]) / 2
        q33 = self.gram[i][2]
        Q = 2 * matrix([[q11, q21, q31], [q12, q22, q32], [q13, q23, q33]])
        p = matrix([0.0, 0.0, 0.0])  # 代表一次项的系数
        G = matrix([[-1.0, 0.0, 0.0], [0.0, -1.0, 0.0], [0.0, 0.0, -1.0]])  # G和h代表GX+s = h，s>=0,表示每一个变量x均大于零
        h = matrix([0.0, 0.0, 0.0])
        A = matrix([1.0, 1.0, 1.0], (1, 3))

        b = matrix(1.0)                                                    # AX = b
        sol = solvers.qp(Q, p, G, h, A, b)

        print(sol['x'])
        for j in range(self.maxk):
            self.W[i][self.neighbor[i][j]] = sol['x'][j]


    def LNPiter(self):
        P = self.W.copy()

        tol = 0.00001     # 误差限
        max_iter = 1000   # 最大迭代次数
        self.num_label_sample = self.num_samples * self.percentage_label_sample
        num_unlabel_sample = self.num_samples - self.num_label_sample

        clamp_data_label = np.zeros((self.num_samples, self.label_num), np.float32)

        for i in xrange(self.num_label_sample):
            # clamp_data_label[i] = self.Y[i]
            if self.Y[i] == -1:
                clamp_data_label[i][0] = 1
            else:
                clamp_data_label[i][1] = 1

        # for i in xrange(num_unlabel_sample):
            # clamp_data_label[i+self.num_label_sample] = 0


        # 分类函数f = Xu
        label_function = clamp_data_label.copy()
        iter_num = 0
        pre_label_function = np.zeros((self.num_samples, self.label_num), np.float32)
        changed = np.abs(pre_label_function - label_function).sum()
        while iter_num < max_iter and changed > tol:
            if iter_num % 1 == 0:
                print "---> Iteration %d/%d, changed: %f" % (iter_num, max_iter, changed)
            pre_label_function = label_function
            iter_num += 1

            # propagation
            label_function = self.alpha * np.dot(P, label_function) + (1-self.alpha) * clamp_data_label

            # check converge
            changed = np.abs(pre_label_function - label_function).sum()

            # get terminate label of unlabeled data

        self.unlabel_data_labels = np.zeros(num_unlabel_sample)
        for i in xrange(num_unlabel_sample):
            if label_function[i + self.num_label_sample][0] > 0:
                self.unlabel_data_labels[i] = -1
            else:
                self.unlabel_data_labels[i] = 1
        
        '''
        correct_num = 0
        for i in xrange(num_unlabel_sample):
            if self.unlabel_data_labels[i] == self.Y[i + self.num_label_sample]:
                correct_num += 1
        print(self.unlabel_data_labels)
        accuracy = correct_num *100/ num_unlabel_sample
        print("Accuracy: %.2f%%" % accuracy)
        '''
        

    def rank_index(self):
        
        A = 0.0
        B = 0.0
        C = 0.0
        D = 0.0
        numSamples = len(self.unlabel_data_labels)
        for i in range(numSamples):
            for j in range(i + 1, numSamples):
                if self.Y[i + self.num_label_sample] == self.Y[j + self.num_label_sample]:
                    if self.unlabel_data_labels[i] == self.unlabel_data_labels[j]:
                        A = A + 1
                    else:
                        B = B + 1
                else:
                    if self.unlabel_data_labels[i] == self.unlabel_data_labels[j]:
                        C = C + 1
                    else:
                        D = D + 1
        print(A, B, C, D)
        accuracy = (A + D) / (A + B + C + D) * 100
        return accuracy



test = LNP()
test.set_filename_and_op("pimaData.txt", ',')
test.read_data()
test.build_graph()
test.set_neighbor()
test.set_gram()
test.solve_weight()
test.LNPiter()
accuracy = test.rank_index()
print("Accuracy: %d%%" %accuracy)
# print(test.getX())