# 遗传算法求解背包问题

# 从配置文件中导入背包容量、物品名称、物品重量、物品价值、物品数量
from typing import Union

from numpy import ndarray

from config import KNAPSACK_CAPACITY, ITEMS_NAMES, ITEMS_WEIGHTS, ITEMS_VALUES, ITEMS_NUM
import numpy as np

from matplotlib import pyplot as plt
import matplotlib as mpl

mpl.use('TkAgg')


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
def generate_initial_population(population_size: int, individual_size: int) -> list[np.ndarray]:
    """
    随机生成一个种群
    :param population_size: 种群大小
    :param individual_size: 染色体长度
    """
    population = []
    for i in range(population_size):
        individual = chromosome_encoding(individual_size)
        population.append(individual)
    return population


# 4. 适应度函数
def fitness_function(individual: np.ndarray) -> int:
    """
    适应度函数
    :param individual: 个体
    """
    # 个体的总价值
    total_value = sum(individual * ITEMS_VALUES)
    # 个体的总重量
    total_weight = sum(individual * ITEMS_WEIGHTS)
    return total_value if total_weight <= KNAPSACK_CAPACITY else 0


# 5. 筛选亲本（轮盘赌）
# 根据适应度计算个体被选中的概率
def calculate_probability(population: list[np.ndarray]) -> np.ndarray:
    """
    计算个体被选中的概率
    :param population: 种群
    """
    fitness = np.array([fitness_function(individual) for individual in population])
    probability = fitness / np.sum(fitness)
    return probability


def roulette_wheel_selection(population: list[np.ndarray], parents_num: int = 0) -> list[np.ndarray]:
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


def tournament_selection(population: list[np.ndarray],
                         tournament_size: int = 3,
                         parents_num: int = 0) -> list[np.ndarray]:
    """
    锦标赛选择
    :param population: 种群
    :param tournament_size: 锦标赛的个体数
    :param parents_num: 选择的父母个数
    """
    parents_num = len(population) if parents_num == 0 else parents_num
    parents = []
    for i in range(parents_num):
        # 随机选择tournament_size个个体
        tournament_individuals = np.random.choice(np.arange(len(population)), size=tournament_size, replace=False)
        tournament_individuals = [population[i] for i in tournament_individuals]
        # 选择适应度最高的个体
        best_individual = select_best_individual(tournament_individuals)
        parents.append(best_individual)
    return parents


def select(population: list, parents_num: int = 0) -> list:
    """
    结合轮盘赌选择和锦标赛选择
    :param population: 种群
    :param parents_num: 选择的父母个数
    """
    parents = []
    parents_num = len(population) if parents_num == 0 else parents_num
    # 轮盘赌选择一半的父母
    parents.extend(roulette_wheel_selection(population, parents_num=parents_num // 2))
    # 锦标赛选择一半的父母
    parents.extend(tournament_selection(population, parents_num=parents_num - parents_num // 2))
    return parents


# 6. 繁殖后代（交叉、变异）
# 交叉
def crossover(parents: list) -> list:
    """
    交叉操作
    :param parents: 父母
    """
    children = []
    for i in range(0, len(parents), 2):
        # 随机选择交叉点
        cross_point = np.random.randint(0, len(parents[i]))
        # 交叉
        child1 = np.concatenate((parents[i][:cross_point], parents[i + 1][cross_point:]))
        child2 = np.concatenate((parents[i + 1][:cross_point], parents[i][cross_point:]))
        children.append(child1)
        children.append(child2)
    return children


# 变异
def mutation(children: list[np.ndarray], mutation_rate: float) -> list[np.ndarray]:
    """
    变异操作
    :param children: 子代
    :param mutation_rate: 变异率
    """
    for i in range(len(children)):
        new_chromosome = np.random.randint(0, 2, size=len(children[i]))
        mutation_index = np.where(np.random.rand(len(children[i])) < mutation_rate)
        children[i][mutation_index] = new_chromosome[mutation_index]
    return children


# 7. 选择下一代
def select_best_individual(population: list) -> np.ndarray:
    """
    选择最优个体
    :param population: 种群
    """
    fitness = np.array([fitness_function(individual) for individual in population])
    best_individual = population[np.argmax(fitness)]
    return best_individual


# 8. 终止条件
def is_termination_condition_met(iteration: int, iteration_times: int = ITERATION_TIMES) -> bool:
    """
    终止条件
    :param iteration: 当前迭代次数
    :param iteration_times: 总迭代次数
    """
    return iteration >= iteration_times


# 9. 主函数
def GeneticAlgorithm(population_size: int = POPULATION_SIZE,
                     iteration_times: int = ITERATION_TIMES,
                     mutation_probability: float = MUTATION_PROBABILITY,
                     mutation_decay: float = 0.99,
                     parents_num_percent: int = 10
                     ) -> np.ndarray:
    """
    遗传算法
    :param population_size: 种群大小
    :param iteration_times: 迭代次数
    :param mutation_probability: 变异概率
    :param mutation_decay: 变异概率衰减率
    :param parents_num_percent: 父母个数百分比
    """
    history_best_fitness = 0
    population = generate_initial_population(population_size, ITEMS_NUM)

    for i in range(iteration_times):
        fitness = np.array([fitness_function(individual) for individual in population])
        best_individual = population[np.argmax(fitness)]
        best_fitness = np.max(fitness)
        history_best_fitness = max(history_best_fitness, best_fitness)
        print(f'第{i + 1}次迭代，最优个体为{best_individual}，最优适应度为{best_fitness}，历史最优适应度为{history_best_fitness}')

        # 去除population中fitness为0的个体
        population = [population[i] for i in range(len(population)) if fitness[i] != 0]

        parents = select(population, parents_num=population_size // parents_num_percent)
        children = crossover(parents)

        mutation_probability *= mutation_decay
        # 绘制在两张图上

        # plt.subplot(121)
        # plt.plot(i + 1, best_fitness, 'ro')
        # plt.xlabel('iteration')
        # plt.ylabel('best fitness')
        # plt.subplot(122)
        # plt.plot(i + 1, mutation_probability, 'bo')
        # plt.xlabel('iteration')
        # plt.ylabel('mutation probability')

        children = mutation(children, MUTATION_PROBABILITY)
        population = children + [best_individual]
        population.extend(generate_initial_population(POPULATION_SIZE - len(population), ITEMS_NUM))
    fitness = np.array([fitness_function(individual) for individual in population])
    best_individual = population[np.argmax(fitness)]
    best_fitness = np.max(fitness)
    print('最优解：', best_individual)
    print('最优解对应的总价值：', best_fitness)
    print('最优解对应的总重量：', sum(best_individual * ITEMS_WEIGHTS))
    print('历史最优解对应的总价值：', history_best_fitness)
    # plt.show()
    return best_individual


if __name__ == '__main__':
    best_individual = GeneticAlgorithm(population_size=10000,
                                       iteration_times=1000,
                                       mutation_probability=0.5,
                                       mutation_decay=1,
                                       parents_num_percent=4
                                       )
