import copy

import numpy as np
import pandas as pd

chessBoardSize = 8
chessBoard = np.zeros((chessBoardSize, chessBoardSize), "byte")


def show_chess_board(chessBoard):
    print("\t", end="")
    for idx in range(chessBoardSize):
        print(idx, end="\t")
    print()
    for row in range(chessBoardSize):
        print(row, end="\t")
        for col in range(chessBoardSize):
            print("♛" if chessBoard[row][col] == 1 else "□", end="\t")
        print()


def show_chess_board_pd(chessBoard):
    res = pd.DataFrame(
        [["■" if chessBoard[row][col] == 1 else "□" for col in range(chessBoardSize)] for row in range(chessBoardSize)])
    print(res)


def show_solution(individual):
    cB = np.zeros((chessBoardSize, chessBoardSize), "byte")
    for row, col in enumerate(individual):
        cB[row][col] = 1

    show_chess_board_pd(cB)


# 初始化个体，每个个体的皇后位置一定是不同行的
# individual = np.random.choice(range(8), 8)
# for row, col in enumerate(individual):
#     chessBoard[row][col] = 1
# show_chess_board_pd(chessBoard)
# print(individual)


def fitness_score(individual):
    """
    information: 计算适应度
    creating time: 2023/3/16
    """
    value = 0
    for queenIndex in range(8):
        for anotherQueenIndex in range(queenIndex + 1, 8):
            if individual[queenIndex] != individual[anotherQueenIndex]:  # 列相同与否
                row_distance = anotherQueenIndex - queenIndex  # 一定大于零
                col_distance = abs(individual[anotherQueenIndex] - individual[queenIndex])
                if row_distance != col_distance:  # 是否处于同一对角线
                    value += 1
    return value


def softmax(fitness_vector):
    """
    information: softmax函数将适应度向量转化为概率分布
    creating time: 2023/3/16
    """
    temp = np.array(fitness_vector, dtype=np.float64)
    return np.exp(temp) / np.exp(temp).sum()


def mutation(individual, prob=0.1):
    """
    information: 随机变异
    creating time: 2023/3/16
    """
    p = np.random.rand(8)
    # print(p)
    individual[p > prob] = np.random.choice(range(8), 8)[p > prob]
    return individual


def GA(size=4):
    """
    information: 遗传算法
    creating time: 2023/3/16
    """

    num_generation = 0  # 进化代数
    population = []  # 一个种群中的个体数量
    for i in range(size):
        # 初始化个体，每个个体的皇后位置一定是不同行的
        population.append(np.random.choice(range(8), 8))

    while True:
        print("Generation : ", num_generation)
        fitness_vector = []  # 种群个体适应度向量
        selection = []

        # 计算每个个体的适应度，以便后面的自然选择
        for individual in population:
            fitness_value = fitness_score(individual)
            if fitness_value == 28:
                print("Find!")
                print(individual)
                return individual
            fitness_vector.append(fitness_value)
        print(fitness_vector)
        print()

        # 自然选择
        prob = softmax(fitness_vector)
        select_id = np.random.choice(range(size), size, replace=True,
                                     p=prob)  # replace=True说明可以有重复项，p=prob表示产生的随机选择要符合prob的概率分布
        for idx in select_id:
            selection.append(population[idx])
        num_pair = int(size / 2)
        position = np.random.choice(range(1, 7), num_pair, replace=True)

        # 开始繁衍，基因交换
        for i in range(0, size, 2):
            start = position[i // 2]
            temp_a = copy.deepcopy(selection[i][start:])
            temp_b = copy.deepcopy(selection[i + 1][start:])

            selection[i][start:] = temp_b
            selection[i + 1][start:] = temp_a

        # 基因变异

        for i in range(size):
            selection[i] = copy.deepcopy(mutation(selection[i], prob=0.8))
        population = selection
        num_generation += 1


# print(np.random.choice(range(4), 4, replace=True, p=[0.1, 0.7, 0.1, 0.1]))
# print(fitness_score(individual))
# print(mutation(individual, prob=0.8))
if __name__ == '__main__':
    show_solution(GA())
