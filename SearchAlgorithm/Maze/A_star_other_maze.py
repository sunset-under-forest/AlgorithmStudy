class Cell(object):
    def __init__(self, x, y, is_wall=False):
        self.x = x
        self.y = y
        self.parent = None
        self.is_wall = is_wall

    # def __eq__(self, other):
    #     return self.x == other.x and self.y == other.y

    def __repr__(self):
        return '({0},{1})'.format(self.x, self.y)

    def get_direction(self, other):
        if self.x == other.x:
            if self.y < other.y:
                return E
            else:
                return W
        else:
            if self.x < other.x:
                return S
            else:
                return N


def get_neighbors(cell):
    neighbors = []
    x, y = cell.x, cell.y
    if x > 0:
        neighbors.append(maze[x - 1][y])
    if x < len(maze) - 1:
        neighbors.append(maze[x + 1][y])
    if y > 0:
        neighbors.append(maze[x][y - 1])
    if y < len(maze[0]) - 1:
        neighbors.append(maze[x][y + 1])
    return neighbors


def a_star(maze, start, end):
    """
    A*算法
    """
    # 排序堆栈
    s = [(0, start)]

    distance_from_start = {start: 0}
    visited = set()
    total_cost = 0
    while s:
        f, cell = s.pop()
        if cell == end:
            # 记录路径
            route = []
            while cell != start:
                route.append(cell)
                cell = cell.parent
            route.append(start)
            route.reverse()

            # 计算总代价
            total_cost = distance_from_start[end]
            return route, total_cost

        if cell not in visited:
            visited.add(cell)

        for neighbor in get_neighbors(cell):
            if neighbor not in visited and not neighbor.is_wall:
                distance_from_start[neighbor] = distance_from_start[cell] + cost[cell.get_direction(neighbor)]
                f = distance_from_start[neighbor]
                s.append((f, neighbor))
                neighbor.parent = cell

        # 按照f值排序
        s.sort(key=lambda x: x[0], reverse=True)

    return False, total_cost


if __name__ == '__main__':
    maze = [
        [1, 1, 1, 1, 1],
        [1, 0, 0, 0, 1],
        [1, 0, 1, 0, 1],
        [1, 0, 1, 1, 1],
        [1, 1, 1, 1, 1]
    ]
    N, S, W, E = ('n', 's', 'w', 'e')
    cost = {N: 5, S: 5, W: 1, E: 1}

    maze = [[Cell(i, j, False if maze[i][j] == 1 else True) for j in range(len(maze[0]))] for i in range(len(maze))]
    print(maze)

    start = maze[2][2]
    end = maze[0][0]
    print("起点：", start)
    print("终点：", end)

    route, total_cost = a_star(maze, start, end)
    print("路径：", route)
    print("总代价：", total_cost)
