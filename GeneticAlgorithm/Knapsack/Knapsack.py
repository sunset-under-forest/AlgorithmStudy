# 遗传算法求解背包问题

# 从配置文件中导入背包容量、物品名称、物品重量、物品价值、物品数量
from typing import Union

from numpy import ndarray

from config import KNAPSACK_CAPACITY, ITEMS_NAMES, ITEMS_WEIGHTS, ITEMS_VALUES, ITEMS_NUM
import numpy as np
import pandas as pd


# 个体类
class Individual(object):
    def __init__(self, chromosome):
        self.chromosome = chromosome  # 染色体
        self.fitness = 0.0  # 适应度
        self.weight = np.sum(self.chromosome * ITEMS_WEIGHTS)  # 重量
        self.value = np.sum(self.chromosome * ITEMS_VALUES)  # 价值

    def __str__(self):
        return 'chromosome: {0}, fitness: {1}, weight: {2}, value: {3}'.format(self.chromosome, self.fitness,
                                                                               self.weight, self.value)


# 1. 对解空间进行编码

# 染色体编码
# 二进制编码
def chromosome_encoding(individual_size: int) -> np.ndarray:
    """
    随机生成一个染色体
    :param individual_size: 染色体长度
    """
    return np.random.randint(0, 2, size=individual_size)


# 2. 设置算法参数

# 种群大小
POPULATION_SIZE = 100
# 迭代次数
ITERATION_TIMES = 1000
# 交叉概率
CROSSOVER_PROBABILITY = 0.8
# 变异概率
MUTATION_PROBABILITY = 0.1


# 3. 初始化种群
def generate_initial_population(population_size: int, individual_size: int) -> list:
    """
    随机生成一个种群
    :param population_size: 种群大小
    :param individual_size: 染色体长度
    """
    population = []
    for i in range(population_size):
        chromosome = chromosome_encoding(individual_size)
        individual = Individual(chromosome)
        population.append(individual)
    return population


# 4. 适应度函数
def fitness_function(individual: Individual) -> Union[int, ndarray]:
    """
    适应度函数
    :param individual: 个体
    """
    if individual.weight > KNAPSACK_CAPACITY:
        return 0
    else:
        return individual.value


# 5. 筛选亲本（轮盘赌）
# 根据适应度计算个体被选中的概率
def calculate_probability(population: list) -> np.ndarray:
    """
    计算个体被选中的概率
    :param population: 种群
    """
    fitness = np.array([fitness_function(individual) for individual in population])
    probability = fitness / np.sum(fitness)
    return probability


def select(population: list) -> list:
    """
    根据适应度计算个体被选中的概率
    :param population: 种群
    """
    probability = calculate_probability(population)
    parents = np.random.choice(population, size=len(population), replace=True, p=probability)
    return parents


# 6. 繁殖后代（交叉、变异）
# 交叉
def crossover(parent1: Individual, parent2: Individual) -> Individual:
    """
    单点交叉
    :param parent1: 亲本1
    :param parent2: 亲本2
    """
    if np.random.rand() < CROSSOVER_PROBABILITY:
        # 选择交叉点
        crossover_point = np.random.randint(0, len(parent1.chromosome))
        # 交叉
        child_chromosome = np.concatenate((parent1.chromosome[:crossover_point], parent2.chromosome[crossover_point:]))
        child = Individual(child_chromosome)
        return child
    else:
        return parent1


# 变异
def mutation(individual: Individual) -> Individual:
    """
    变异
    :param individual: 个体
    """
    for gene in range(len(individual.chromosome)):
        if np.random.rand() < MUTATION_PROBABILITY:
            individual.chromosome[gene] ^= 1 # 异或运算，相当于取反
    return individual


if __name__ == '__main__':
    # 初始化种群
    population = generate_initial_population(POPULATION_SIZE, ITEMS_NUM)
    print('初始种群：')
    for idx, individual in enumerate(population):
        print(idx, individual)
