# 记录背包问题的相关配置信息

# 背包容量
KNAPSACK_CAPACITY = 6404180

# 读取物品信息
import pandas as pd
items = pd.read_csv('knapsack_items.csv')
# print(items)

# 物品数量
ITEMS_NUM = len(items)

# 物品名称
ITEMS_NAMES = items['name'].values

# 物品重量
ITEMS_WEIGHTS = items['weight'].values

# 物品价值
ITEMS_VALUES = items['value'].values
