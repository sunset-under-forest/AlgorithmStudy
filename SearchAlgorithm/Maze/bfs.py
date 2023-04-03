from collections import deque

# 按照上面的代码，写出一个记录路径的bfs函数
def bfs(maze, start, end):
    """
    广度优先搜索
    :param maze: maze对象
    :param start: cell对象，起点
    :param end: cell对象，终点
    """
    queue = deque()
    queue.append(start)
    visited = set()
    visited.add(start)
    # 记录路径
    path = {start: None}


    while queue:
        cell = queue.popleft()
        if cell == end:
            # 记录路径
            route = []
            while cell != start:
                route.append(cell)
                cell = path[cell]
            route.append(start)
            route.reverse()
            return route

        # 按上、下、左、右顺序搜索相邻节点
        for neighbor in maze.neighbors(cell):
            # 判断节点是否在迷宫内，并且没有被访问过，并且与当前节点相连
            if neighbor not in visited and cell.is_connected(neighbor):
                path[neighbor] = cell
                queue.append(neighbor)
                visited.add(neighbor)

    return False