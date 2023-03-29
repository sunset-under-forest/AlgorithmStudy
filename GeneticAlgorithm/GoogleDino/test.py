import numpy as np


def softmax(fitness_vector):
    """
    information: softmax函数将适应度向量转化为概率分布
    creating time: 2023/3/16
    """
    temp = np.array(fitness_vector, dtype=np.float64)
    return np.exp(temp) / np.exp(temp).sum()


scores = [15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15,
          15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15,
          15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15,
          15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15,
          15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15,
          15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15,
          15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16,
          16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16,
          16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16,
          16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16,
          16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16,
          16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16,
          16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16,
          16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16,
          16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16,
          16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16,
          16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 34, 34, 34, 34, 34, 34, 34, 34, 34,
          34, 34, 34, 91, 91, 92, 92, 92, 92, 92, 92, 92, 92, 92, 92, 92, 92, 92, 92, 92, 92, 92, 92, 92, 113, 131, 131,
          131, 131, 131, 131, 131, 154, 189, 189, 189, 246, 408, 408, 571, 571]
scores = np.array(scores, dtype=np.float64)
# 精英数量
elite_size = 10
# 将scores后面10个设为精英，单独提出来
strong_individuals = scores[-elite_size:]
normal_individuals = scores[:-elite_size]


# 标准化函数
def standardize(x):
    x_std = (x - x.mean()) / x.std()
    return x_std


# 归一化函数
def normalize(x):
    # 判断是否全部一样
    if x.max() == x.min():
        x_norm = np.ones_like(x)
    else:

        x_norm = (x - x.min()) / (x.max() - x.min())
    return x_norm


# scores = standardize(scores)
# print(normal_individuals)
# print(strong_individuals)
# print(normalize(normal_individuals).sum())
# print(normalize(strong_individuals) .sum())
#
# # 为精英分配较大的概率，合为一个数组
# scores = np.concatenate((normalize(normal_individuals) * 0.5 , normalize(strong_individuals) * 0.5) )

# scores = normalize(scores)
#
# print(scores , sum(scores))
# prob = np.concatenate((softmax(normal_individuals) * 0.5 , softmax(strong_individuals) * 0.5))
# print(prob,sum(prob))
# print(prob[-10:],sum(prob[-10:]))

# strong_individuals = np.array([ 379, 379, 379, 379, 379, 379, 379, 379, 379, 379])
# strong_individuals = normalize(strong_individuals)
# print(softmax(strong_individuals))

# with open("test.npz", "wb") as f:
#     np.save(f, scores)
#
# with open("test.npz", "rb") as f:
#     scores = np.load(f)
#     print(scores)