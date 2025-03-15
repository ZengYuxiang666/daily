# 这题是记忆化搜索
# https://leetcode.cn/problems/count-paths-with-the-given-xor-value
from functools import cache
from typing import List


class Solution:
    def countPathsWithXorValue(self, grid: List[List[int]], k: int) -> int:
        MOD = 1_000_000_007  # 取模，防止结果过大

        @cache  # 使用 Python 的缓存机制，避免重复计算
        def dfs(i: int, j: int, x: int) -> int:
            """
            递归计算从 (0,0) 到 (i,j) 的路径数，使得 XOR 值为 x。

            参数：
            i: 当前行索引
            j: 当前列索引
            x: 目标 XOR 值

            返回：
            满足 XOR 值等于 x 的路径总数
            """
            if i < 0 or j < 0:
                return 0  # 越界情况，返回 0（无效路径）

            val = grid[i][j]  # 当前格子的值

            if i == 0 and j == 0:
                return 1 if x == val else 0  # 到达起点时检查 XOR 值是否匹配

            # 计算两种可能路径：
            # 1. 从左边的格子 (i, j-1) 过来
            # 2. 从上面的格子 (i-1, j) 过来
            # 在计算时，把当前格子的值 `val` 与目标 XOR 值 `x` 进行异或操作
            return (dfs(i, j - 1, x ^ val) + dfs(i - 1, j, x ^ val)) % MOD

            # 计算从 (0,0) 到 (m-1,n-1) 的路径数，使 XOR 结果等于 k

        return dfs(len(grid) - 1, len(grid[0]) - 1, k)
