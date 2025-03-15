# 与边界有关，尤其是与边界接触就不算的可以先遍历边界，把他全变成海，然后再遍历里面的
# https://leetcode.cn/problems/number-of-closed-islands
class Solution:
    def closedIsland(self, grid):
        move_x = [1, -1, 0, 0]  # 上下左右方向
        move_y = [0, 0, 1, -1]  # 上下左右方向

        # 辅助函数：DFS遍历
        def dfs(x, y):
            # 标记当前为访问过的区域
            grid[x][y] = 1
            # 继续向四个方向扩展
            for i in range(4):
                x1, y1 = x + move_x[i], y + move_y[i]
                if 0 <= x1 < len(grid) and 0 <= y1 < len(grid[0]) and grid[x1][y1] == 0:
                    dfs(x1, y1)

        # 1. 从四个边界进行DFS，标记所有与边界相连的0为访问过
        for i in range(len(grid)):
            if grid[i][0] == 0:  # 第一列
                dfs(i, 0)
            if grid[i][len(grid[0]) - 1] == 0:  # 最后一列
                dfs(i, len(grid[0]) - 1)

        for j in range(len(grid[0])):
            if grid[0][j] == 0:  # 第一行
                dfs(0, j)
            if grid[len(grid) - 1][j] == 0:  # 最后一行
                dfs(len(grid) - 1, j)

        # 2. 计算封闭岛屿的数量
        total = 0
        for i in range(1, len(grid) - 1):  # 避开边界
            for j in range(1, len(grid[0]) - 1):  # 避开边界
                if grid[i][j] == 0:  # 找到一个未被访问过的陆地
                    total += 1
                    dfs(i, j)  # 启动DFS，标记整个岛屿

        return total

