# 求一个格子到另一类格子的最短距离，可以在开始时将一类格子全放入queue中
# https://leetcode.cn/problems/as-far-from-land-as-possible
from collections import deque
class Solution(object):
    def maxDistance(self, grid):
        n = len(grid)

        # 检查是否全是陆地或全是海洋
        if all(grid[i][j] == 1 for i in range(n) for j in range(n)):
            return -1
        if all(grid[i][j] == 0 for i in range(n) for j in range(n)):
            return -1

        # BFS 初始化
        queue = deque()
        visited = [[False] * n for _ in range(n)]

        # 将所有陆地单元格加入队列
        for i in range(n):
            for j in range(n):
                if grid[i][j] == 1:
                    queue.append((i, j, 0))  # (x, y, distance)
                    visited[i][j] = True

        # 四个方向：上下左右
        directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]

        # BFS 扩展
        max_distance = -1
        while queue:
            x, y, dist = queue.popleft()

            # 遍历四个方向
            for dx, dy in directions:
                nx, ny = x + dx, y + dy
                if 0 <= nx < n and 0 <= ny < n and not visited[nx][ny]:
                    if grid[nx][ny] == 0:  # 如果是海洋
                        visited[nx][ny] = True
                        queue.append((nx, ny, dist + 1))
                        max_distance = max(max_distance, dist + 1)

        return max_distance