# 对访问过的点用集合储存可以避免潜在错误
# https://leetcode.cn/problems/shortest-path-in-binary-matrix
from collections import deque


class Solution(object):
    def shortestPathBinaryMatrix(self, grid):
        n = len(grid)

        # 如果起点或终点是 1，直接返回 -1
        if grid[0][0] == 1 or grid[n - 1][n - 1] == 1:
            return -1

        # 定义 8 个方向（上下左右及四个对角线方向）
        directions = [(-1, 0), (1, 0), (0, -1), (0, 1), (-1, -1), (-1, 1), (1, -1), (1, 1)]

        # 队列存储当前的 (x, y) 位置以及路径长度
        queue = deque([(0, 0, 1)])  # 起点 (0, 0) 的路径长度为 1
        visited = set([(0, 0)])  # 记录访问过的点

        while queue:
            x, y, dist = queue.popleft()

            # 如果到达终点 (n-1, n-1)，返回当前路径长度
            if (x, y) == (n - 1, n - 1):
                return dist

            # 尝试所有 8 个方向
            for dx, dy in directions:
                nx, ny = x + dx, y + dy

                # 检查新位置是否在矩阵内，且值为 0 且未访问过
                if 0 <= nx < n and 0 <= ny < n and grid[nx][ny] == 0 and (nx, ny) not in visited:
                    queue.append((nx, ny, dist + 1))
                    visited.add((nx, ny))

        # 如果遍历完队列仍未找到路径，返回 -1
        return -1
