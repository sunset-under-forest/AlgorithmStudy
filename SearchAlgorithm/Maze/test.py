from collections import deque

# 定义迷宫地图和起点、终点坐标
maze = [
    [1, 1, 1, 1, 1],
    [1, 0, 0, 0, 1],
    [1, 1, 1, 0, 1],
    [1, 0, 0, 0, 1],
    [1, 1, 1, 1, 1]
]
start = (1, 1)
end = (3, 3)

# 定义广度优先搜索函数
def bfs(maze, start, end):
    queue = deque()
    queue.append(start)
    visited = set()
    visited.add(start)

    while queue:
        x, y = queue.popleft()
        if (x, y) == end:
            return True

        # 按上、下、左、右顺序搜索相邻节点
        for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            nx, ny = x + dx, y + dy
            # 判断节点是否在迷宫内，并且没有被访问过，并且不是障碍物
            if 0 <= nx < len(maze) and 0 <= ny < len(maze[0]) and (nx, ny) not in visited and maze[nx][ny] == 0:
                queue.append((nx, ny))
                visited.add((nx, ny))

    return False

# 调用广度优先搜索函数，并输出结果
if bfs(maze, start, end):
    print("找到终点！")
else:
    print("未找到终点。")
