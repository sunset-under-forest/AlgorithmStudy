def dfs(maze, start, end):
    """
    深度优先搜索
    :param maze: maze对象
    :param start: cell对象，起点
    :param end: cell对象，终点
    """
    stack = [start]
    visited = set()
    visited.add(start)
    while stack:
        node = stack[-1]
        if node == end:
            return stack
        if node not in visited:
            visited.add(node)

        find = False
        for neighbor in maze.neighbors(node):
            if neighbor not in visited and node.is_connected(neighbor):
                stack.append(neighbor)
                find = True
                break
        if not find:
            stack.pop()

    return False
