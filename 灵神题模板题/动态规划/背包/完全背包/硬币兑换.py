# dp的题目在开始之前应该先初始化dp数组，比如dp[0]这种特殊的
# 给你一个数组要你去算的应该想到是dp
# https://leetcode.cn/problems/coin-change-ii
class Solution(object):
    def change(self, amount, coins):
        dp = [0]*(amount+1)
        dp[0] = 1
        for coin in coins:
            for v in range(coin,amount+1):
                dp[v] += dp[v-coin]
        return dp[amount]