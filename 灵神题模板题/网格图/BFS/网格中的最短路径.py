# 病毒扩散式搜索，用三维数组判断要不要走这个格子，最先达到的肯定是最快的
# https://leetcode.cn/problems/shortest-path-in-a-grid-with-obstacles-elimination
class Solution(object):
    def shortestPath(self, grid, k):
        from collections import deque
        m, n = len(grid), len(grid[0])

        # 如果没有障碍物限制，直接用曼哈顿距离
        if k >= m + n - 2:
            return m + n - 2

        directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]
        visited = [[[False] * (k + 1) for _ in range(n)] for _ in range(m)]
        queue = deque([(0, 0, k, 0)])  # (x, y, remaining_k, steps)
        visited[0][0][k] = True

        while queue:
            x, y, remaining_k, steps = queue.popleft()

            # 到达终点
            if (x, y) == (m - 1, n - 1):
                return steps

            # 扩展四个方向
            for dx, dy in directions:
                nx, ny = x + dx, y + dy
                if 0 <= nx < m and 0 <= ny < n:
                    new_k = remaining_k - grid[nx][ny]
                    if new_k >= 0 and not visited[nx][ny][new_k]:
                        visited[nx][ny][new_k] = True
                        queue.append((nx, ny, new_k, steps + 1))

        return -1