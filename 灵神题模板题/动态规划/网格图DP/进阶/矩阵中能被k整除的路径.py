# 因为k的取值范围小，所以可以创建一个三维的dp数组，dp[i][j][rem]为起点到(i,j)点的路径和余k为rem的路径数量
# 初始化两边在哪里呢？ 状态转移方程中的if i > 0和if j > 0:解决了两边要单独求的问题
# https://leetcode.cn/problems/paths-in-matrix-whose-sum-is-divisible-by-k
from typing import List
class Solution:
    def numberOfPaths(self, grid: List[List[int]], k: int) -> int:
        MOD = 10 ** 9 + 7
        m, n = len(grid), len(grid[0])

        # 三维 DP 数组 (m x n x k)
        dp = [[[0] * k for _ in range(n)] for _ in range(m)]

        # 初始化起点
        dp[0][0][grid[0][0] % k] = 1

        # 遍历 DP 表
        for i in range(m):
            for j in range(n):
                for rem in range(k):
                    if i > 0:
                        dp[i][j][(rem + grid[i][j]) % k] += dp[i - 1][j][rem]
                        dp[i][j][(rem + grid[i][j]) % k] %= MOD

                    if j > 0:
                        dp[i][j][(rem + grid[i][j]) % k] += dp[i][j - 1][rem]
                        dp[i][j][(rem + grid[i][j]) % k] %= MOD

        # 目标点 (m-1, n-1) 余数为 0 的路径数
        return dp[m - 1][n - 1][0]