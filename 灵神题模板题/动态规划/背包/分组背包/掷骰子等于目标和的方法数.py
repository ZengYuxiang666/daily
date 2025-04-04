# https://leetcode.cn/problems/number-of-dice-rolls-with-target-sum
class Solution(object):
    def numRollsToTarget(self, n, k, target):
        MOD = 10 ** 9 + 7
        # 初始化DP表
        dp = [[0] * (target + 1) for _ in range(n + 1)]
        dp[0][0] = 1

        # 动态规划计算
        for i in range(1, n + 1):  # 骰子个数
            for j in range(1, target + 1):  # 和为 j
                for x in range(1, k + 1):  # 当前骰子面数
                    if j >= x:
                        dp[i][j] = (dp[i][j] + dp[i - 1][j - x]) % MOD

        return dp[n][target]
