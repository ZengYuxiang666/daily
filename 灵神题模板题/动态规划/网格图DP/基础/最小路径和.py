"""
Dijkstra 可以解这个问题，但不如 DP 高效。
DP 是更优解，尤其适用于**方向固定（只能右、下）**的网格问题。
"""
# https://leetcode.cn/problems/minimum-path-sum
class Solution(object):
    def minPathSum(self, grid):
        m, n = len(grid), len(grid[0])
        dp = [[0] * n for _ in range(m)]

        # 初始化起点
        dp[0][0] = grid[0][0]

        # 初始化第一列
        for i in range(1, m):
            dp[i][0] = dp[i-1][0] + grid[i][0]

        # 初始化第一行
        for j in range(1, n):
            dp[0][j] = dp[0][j-1] + grid[0][j]

        # 计算 DP 表
        for i in range(1, m):
            for j in range(1, n):
                dp[i][j] = grid[i][j] + min(dp[i-1][j], dp[i][j-1])

        return dp[m-1][n-1]