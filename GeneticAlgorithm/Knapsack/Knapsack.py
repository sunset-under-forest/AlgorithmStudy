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
POPULATION_SIZE = 1000
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
    return individual.value if individual.weight <= KNAPSACK_CAPACITY else 0


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


def roulette_wheel_selection(population: list, parents_num: int = 0) -> list:
    """
    轮盘赌选择
    :param population: 种群
    :param parents_num: 选择的父母个数
    """
    parents_num = len(population) if parents_num == 0 else parents_num
    probability = calculate_probability(population)
    parents = np.random.choice(np.arange(len(population)), size=parents_num, replace=True, p=probability)
    parents = [population[i] for i in parents]
    return parents


# 6. 繁殖后代（交叉、变异）
# 交叉
def crossover(parent1: Individual, parent2: Individual,
              crossover_probability: float = 1) -> Individual:
    """
    交叉
    :param parent1: 亲本1
    :param parent2: 亲本2
    :param crossover_probability: 交叉概率
    """
    if np.random.rand() < crossover_probability:
        point = np.random.randint(0, len(parent1.chromosome))
        child = Individual(np.concatenate((parent1.chromosome[:point], parent2.chromosome[point:])))
    else:
        child = Individual(parent1.chromosome)
    return child


# 变异
def mutation(individual: Individual, mutation_probability: float = 1) -> Individual:
    """
    变异
    :param individual: 个体
    :param mutation_probability: 变异概率
    """
    for gene in range(len(individual.chromosome)):
        if np.random.rand() < mutation_probability:
            individual.chromosome[gene] ^= 1  # 异或运算，相当于取反
    return individual


# 7. 选择下一代
def next_generation(population: list, crossover_probability: float, mutation_probability: float) -> list:
    """
    选择下一代
    :param population: 种群
    :param crossover_probability: 交叉概率
    :param mutation_probability: 变异概率
    """
    parents = roulette_wheel_selection(population)
    next_generation = []
    for i in range(0, len(parents), 2):
        parent1 = parents[i]
        parent2 = parents[i + 1]
        child1 = crossover(parent1, parent2, crossover_probability)
        child2 = crossover(parent2, parent1, crossover_probability)
        child1 = mutation(child1, mutation_probability)
        child2 = mutation(child2, mutation_probability)
        next_generation.append(child1)
        next_generation.append(child2)
    return next_generation


# 8. 终止条件
def is_termination_condition_met(iteration: int, iteration_times: int = ITERATION_TIMES) -> bool:
    """
    终止条件
    :param iteration: 当前迭代次数
    :param iteration_times: 总迭代次数
    """
    return iteration >= iteration_times


# 9. 主函数
def GeneticAlgorithm(population_size: int, iteration_times: int, crossover_probability: float,
                     mutation_probability: float) -> Individual:
    """
    遗传算法
    :param population_size: 种群大小
    :param iteration_times: 迭代次数
    :param crossover_probability: 交叉概率
    :param mutation_probability: 变异概率
    """
    # 初始化种群
    population = generate_initial_population(population_size, ITEMS_NUM)
    # 迭代
    iteration = 0

    # 随迭代次数增加，突变概率减少
    mutation_probability = 1 - (1 - mutation_probability) * iteration / iteration_times
    best_fitness = 0
    while best_fitness != 13692887:
        population = next_generation(population, crossover_probability, mutation_probability)
        # 计算最大适应度
        fitness = [fitness_function(individual) for individual in population]
        best_fitness = np.max(fitness)
        print('iteration: {0}, max fitness: {1}'.format(iteration, best_fitness))
        iteration += 1
    # 计算适应度
    fitness = [fitness_function(individual) for individual in population]
    # 选择最优个体
    best_individual = population[np.argmax(fitness)]
    return best_individual


if __name__ == '__main__':
    best_individual = GeneticAlgorithm(POPULATION_SIZE, ITERATION_TIMES, CROSSOVER_PROBABILITY, MUTATION_PROBABILITY)
    print(best_individual)
