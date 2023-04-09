# 遗传算法求解背包问题

# 从配置文件中导入背包容量、物品名称、物品重量、物品价值、物品数量
from config import KNAPSACK_CAPACITY, ITEMS_NAMES, ITEMS_WEIGHTS, ITEMS_VALUES, ITEMS_NUM
import numpy as np
import pandas as pd


# 个体类
class Individual(object):
    def __init__(self, chromosome):
        self.chromosome = chromosome  # 染色体
        self.fitness = 0.0  # 适应度
        self.weight = 0.0  # 重量
        self.value = 0.0  # 价值

    def __str__(self):
        return 'chromosome: {0}, fitness: {1}, weight: {2}, value: {3}'.format(self.chromosome, self.fitness,
                                                                               self.weight, self.value)


# 染色体编码
# 二进制编码
def chromosome_encoding(size:int) -> np.ndarray:
    """
    随机生成一个染色体
    :param size: 染色体长度
    """
    return np.random.randint(0, 2, size=size)

if __name__ == '__main__':
    print(chromosome_encoding(10))