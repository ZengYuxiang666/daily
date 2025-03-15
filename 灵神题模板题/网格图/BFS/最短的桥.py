# 多路实现病毒式扩散
# https://leetcode.cn/problems/shortest-bridge
from collections import deque
class Solution(object):
    def shortestBridge(self, grid):
        n = len(grid)
        directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]

        def find_first_island():
            """找到第一个岛屿并返回其所有坐标，同时将其标记为访问过"""
            for i in range(n):
                for j in range(n):
                    if grid[i][j] == 1:
                        queue = deque([(i, j)])
                        grid[i][j] = -1  # 标记为访问过
                        island = []
                        while queue:
                            x, y = queue.popleft()
                            island.append((x, y))
                            for dx, dy in directions:
                                nx, ny = x + dx, y + dy
                                if 0 <= nx < n and 0 <= ny < n and grid[nx][ny] == 1:
                                    queue.append((nx, ny))
                                    grid[nx][ny] = -1  # 标记为访问过
                        return island

        # 第一步：找到第一个岛屿
        island1 = find_first_island()

        # 第二步：多源 BFS，从第一个岛屿扩展，寻找第二个岛屿
        queue = deque(island1)
        steps = 0

        while queue:
            for _ in range(len(queue)):  # bfs关键
                x, y = queue.popleft()
                for dx, dy in directions:
                    nx, ny = x + dx, y + dy
                    if 0 <= nx < n and 0 <= ny < n:
                        if grid[nx][ny] == 1:  # 找到第二个岛屿
                            return steps
                        if grid[nx][ny] == 0:  # 扩展到水域
                            grid[nx][ny] = -1  # 标记为访问过
                            queue.append((nx, ny))
            steps += 1

        return -1  # 不应该到这里

