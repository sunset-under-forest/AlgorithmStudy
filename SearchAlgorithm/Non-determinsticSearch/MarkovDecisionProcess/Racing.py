"""
马尔可夫决策过程，赛车问题
问题描述：
一辆由你写的机器人汽车在路上行驶，想要尽可能地开的快，但是加速可能会导致过热
它有三个状态：低温(cool)、高温(warm)和过热(overheat)
它有两个动作：加速(fast)和减速(slow)
如果汽车处于过热状态，则汽车损毁，程序结束
已知的转移概率如下：
T(cool, fast, cool) = 0.5
T(cool, fast, warm) = 0.5
T(cool, fast, overheat) = 0
T(cool, slow, cool) = 1
T(cool, slow, warm) = 0
T(cool, slow, overheat) = 0
T(warm, fast, cool) = 0
T(warm, fast, warm) = 0.5
T(warm, fast, overheat) = 0.5
T(warm, slow, cool) = 0
T(warm, slow, warm) = 0.5
T(warm, slow, overheat) = 0.5

已知的奖励如下：
R(cool, fast, cool) = 2
R(cool, fast, warm) = 2
R(cool, slow, cool) = 1
R(warm, slow, cool) = 1
R(warm, slow, warm) = 1
R(warm, fast, warm) = 2
R(warm, fast, overheat) = -10

起始状态为cool，终止状态为overheat，目标是使用MDP算法找到最优策略，即处于每个状态时应该采取的最佳动作
"""
import numpy as np

COOL, WARM, OVERHEAT = 0, 1, 2
SLOW, FAST = 0, 1

# 状态集合
STATES = [COOL, WARM, OVERHEAT]
STATES_NAME = ["COOL", "WARM", "OVERHEAT"]
# 动作集合
ACTIONS = [SLOW, FAST]
ACTIONS_NAME = ["SLOW", "FAST"]

# 转移概率矩阵，3x2x3，第一维表示当前状态，第二维表示动作，第三维表示下一个状态，值表示概率
TRANSITION_PROBABILITY = np.array([
    # cool
    [[1, 0, 0],  # slow
     [0.5, 0.5, 0]],  # fast
    # warm
    [[0.5, 0.5, 0],  # slow
     [0, 0, 1]],  # fast
    # overheat
    [[0, 0, 0],  # slow
     [0, 0, 0]],  # fast
])

# 奖励矩阵，3x2x3，第一维表示当前状态，第二维表示动作，第三维表示下一个状态，值表示奖励
REWARD = np.array([
    # cool
    [[1, 0, 0],  # slow
     [2, 2, 0]],  # fast
    # warm
    [[1, 1, 0],  # slow
     [0, 0, -10]],  # fast
    # overheat
    [[0, 0, 0],  # slow
     [0, 0, 0]],  # fast
])


# 因为OVERHEAT是终止状态，所以转移概率和奖励都是0

# 结束状态
def is_terminal(state: int) -> bool:
    """
    判断是否是结束状态
    :param state: 状态
    :return: 是否是结束状态
    """
    return state == OVERHEAT


# 状态转移函数
def transition(state: int, action: int) -> int:
    """
    状态转移函数
    :param state: 当前状态
    :param action: 动作
    :return: 下一个状态
    """
    return np.random.choice(STATES, p=TRANSITION_PROBABILITY[state, action])


# 奖励函数
def reward(state: int, action: int, next_state: int) -> int:
    """
    奖励函数
    :param state: 当前状态
    :param action: 动作
    :param next_state: 下一个状态
    :return: 奖励
    """
    return REWARD[state, action, next_state]


def main():
    # Optimal Quantities
    # 最优状态价值函数
    V = np.zeros(len(STATES))  # 代表每个状态的价值，行代表状态
    # 最优动作价值函数
    Q = np.zeros((len(STATES), len(ACTIONS)))  # 代表每个状态下的每个动作的价值，行代表状态，列代表动作
    # 最优策略
    policy = np.zeros(len(STATES), dtype=int)  # 代表每个状态下的最优动作，行代表状态，值代表动作

    # Recursive definition of optimal value functions
    # 递归定义最优状态价值函数和最优动作价值函数
    # V(s) = max_a Q(s, a)
    # Q(s, a) = sum_s'(T(s, a, s') * (R(s, a, s') + gamma * V(s')))
    # V(s) = max_a sum_s'(T(s, a, s') * (R(s, a, s') + gamma * V(s')))

    MAX_ITERATION = 50  # 最大迭代次数
    GAMMA = 0.5  # 折扣因子

    # Value Iteration
    # 价值迭代
    for i in range(MAX_ITERATION):
        print("第{}次迭代，V = {}".format(i, V))
        # Update Q(s, a) = sum_s'(T(s, a, s') * (R(s, a, s') + gamma * V(s')))
        for state in STATES:
            for action in ACTIONS:
                Q[state, action] = 0  # 每一次都要重新计算
                for next_state in STATES:
                    Q[state, action] += TRANSITION_PROBABILITY[state, action, next_state] * (
                            REWARD[state, action, next_state] + GAMMA * V[next_state])

        # Update V(s) = max_a Q(s, a)
        V = np.max(Q,
                   axis=1)  # axis=1表示按行取最大值，因为式子V(s) = max_a Q(s, a)代表在每个状态s下取不同动作a的最大值，而Q是3x2的矩阵，行代表状态s，列代表动作a，所以axis=1表示按行（每个状态s）取最大值（列a是变量）

        # 根据最优状态价值函数更新最优策略
        policy = np.argmax(Q, axis=1)

    # Print optimal policy
    # 打印最优策略
    print("最优策略：")
    for state in STATES[:-1]:
        print("{}: {}".format(STATES_NAME[state], ACTIONS_NAME[policy[state]]))

    print("V(s): {}".format(V))


if __name__ == '__main__':
    main()
