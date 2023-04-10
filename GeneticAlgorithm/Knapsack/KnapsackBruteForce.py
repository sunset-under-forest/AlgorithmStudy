# 蛮力法求解背包问题

# 从配置文件中导入背包容量、物品名称、物品重量、物品价值、物品数量
from config import KNAPSACK_CAPACITY, ITEMS_NAMES, ITEMS_WEIGHTS, ITEMS_VALUES, ITEMS_NUM

import numpy as np
import time

# 物品选择列表，二进制编码
# 0表示不选，1表示选
items_select = np.zeros(ITEMS_NUM, dtype=np.int8)
print(items_select, items_select.shape)

# 最好的选择情况 [物品选择列表，总重量，总价值]
best_items_select = [items_select, 0, 0]

# 计时
start_time = time.time()
# 遍历所有可能的物品选择情况
# 2^ITEMS_NUM 种情况
for i in range(2 ** ITEMS_NUM):
    # 二进制转换为十进制
    # 二进制的第i位表示第i个物品是否选择
    # 例如：i=5，二进制为101，表示第0、2个物品被选择
    items_select = np.array(list(np.binary_repr(i, ITEMS_NUM)), dtype=np.int8)
    # 计算总重量
    total_weight = np.sum(items_select * ITEMS_WEIGHTS)
    # 计算总价值
    total_value = np.sum(items_select * ITEMS_VALUES)
    # 如果总重量小于背包容量
    if total_weight <= KNAPSACK_CAPACITY:
        # 如果总价值大于之前的最大总价值
        if total_value > best_items_select[2]:
            # 更新最好的选择情况
            best_items_select = [items_select, total_weight, total_value]
            print(i, best_items_select)

# 计时结束
end_time = time.time()
print('最好的选择情况：', best_items_select)
print('总共耗时：', end_time - start_time)

