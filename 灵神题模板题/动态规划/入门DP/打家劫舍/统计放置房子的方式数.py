# 第i个位置只能放放房子或者不放房子，当放房子时的方案数为dp[i-2]，不放房子时为dp[i-1],当这两个相加时才是第i个位置的最大方案数
# https://leetcode.cn/problems/count-number-of-ways-to-place-houses
class Solution(object):
    def countHousePlacements(self, n):
        MOD = 10 ** 9 + 7
        dp = [0] * (n + 1)
        dp[0] = 1  # 0个地块时只有1种方案（不放置）
        dp[1] = 2  # 1个地块时有2种方案（放房子 or 空）

        for i in range(2, n + 1):
            dp[i] = (dp[i - 1] + dp[i - 2]) % MOD

        return (dp[n] ** 2) % MOD  # 结果平方
