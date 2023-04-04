# A*算法求解路径
# 先生成一份迷宫地图


from MazeGenerator import *

# 生成迷宫地图
maze = Maze.generate(10, 5)

# 生成起点和终点
start = maze.get_random_cell()
target = maze.get_random_cell()

print(maze)
print("起点：", start)
print("终点：", target)

# 朝各个方向移动的代价
cost = {N: 5, S: 5, W: 1, E: 1}

# 总代价
total_cost = 0


# A*算法求解路径
def a_star(maze, start, end):
    """
    A*算法求解路径
    :param maze: maze对象
    :param start: cell对象，起点
    :param end: cell对象，终点
    """
    global total_cost
    # 优先队列，按照f值排序
    s = [(0, start)]
    # 记录路径
    path = {start: None}

    distance_from_start = {start: 0}
    visited = set()

    while s:
        f, cell = s.pop()
        if cell == end:
            # 记录路径
            route = []
            while cell != start:
                route.append(cell)
                cell = path[cell]
            route.append(start)
            route.reverse()
            # 计算总代价
            total_cost = distance_from_start[end]
            return route

        if cell not in visited:
            visited.add(cell)

        for neighbor in maze.neighbors(cell):
            if neighbor not in visited and cell.is_connected(neighbor):
                distance_from_start[neighbor] = distance_from_start[cell] + cost[cell.get_direction(neighbor)]
                f = distance_from_start[neighbor]
                s.append((f, neighbor))
                path[neighbor] = cell

        # 按照f值排序
        s.sort(key=lambda x: x[0])

    return False


if __name__ == '__main__':
    import sys

    if len(sys.argv) > 1:
        width = int(sys.argv[1])
        if len(sys.argv) > 2:
            height = int(sys.argv[2])
        else:
            height = width
    else:
        width = 10
        height = 5

    try:
        while True:
            game = MazeGame(Maze.generate(width, height))
            if not game.play(a_star):
                print("Total cost: ", total_cost)
                break
    except:
        import traceback

        traceback.print_exc(file=open('error_log.txt', 'a'))
