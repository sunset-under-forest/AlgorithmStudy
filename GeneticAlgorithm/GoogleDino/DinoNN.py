import copy

import numpy as np


# 将上面的类转化为一个两层的神经网络
class DinoNN:
    def __init__(self):
        # 初始化权重矩阵，4个输入3个输出
        # 隐藏层7个神经元
        self.hidden_layer = 7
        self.weights1 = 2 * np.random.random((4, self.hidden_layer)) - 1
        self.weights2 = 2 * np.random.random((self.hidden_layer, 3)) - 1
        self.fitness = 0

    # 两个神经网络的基因杂交，杂交按概率不同网络的基因赋值权重，最后相加
    def cross(self, other, prob=0.5):
        # 杂交
        # 杂交权重
        weights1 = self.weights1 * prob + other.weights1 * (1 - prob)
        weights2 = self.weights2 * prob + other.weights2 * (1 - prob)

        # 生成新的神经网络
        new_nn = DinoNN()
        new_nn.weights1 = weights1
        new_nn.weights2 = weights2

        return new_nn

    def mutation(self, prob=0.2):
        # 变异
        # 对权重的每一个值进行变异
        for i in range(self.weights1.shape[0]):
            for j in range(self.weights1.shape[1]):
                if np.random.random() < prob:
                    self.weights1[i][j] = 2 * np.random.random() - 1
        for i in range(self.weights2.shape[0]):
            for j in range(self.weights2.shape[1]):
                if np.random.random() < prob:
                    self.weights2[i][j] = 2 * np.random.random() - 1

    def sigmoid(self, x):
        # 激活函数
        return 1 / (1 + np.exp(-x))

    def sigmoid_derivative(self, x):
        # 激活函数的导数
        return x * (1 - x)

    def predict(self, inputs):
        # 预测
        output = self.sigmoid(np.dot(inputs, self.weights1))
        output = self.sigmoid(np.dot(output, self.weights2))
        # 使用softmax函数
        exp_output = np.exp(output)
        return exp_output / np.sum(exp_output, axis=0, keepdims=True)

    def save(self, file):
        # 保存权重
        np.save(file, self.weights1)
        np.save(file, self.weights2)
