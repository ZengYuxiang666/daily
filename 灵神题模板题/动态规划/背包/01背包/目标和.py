# 创建一个sum(nums)的dp数组，dp[i]表示组合为i有dp[i]种方法，状态转移方程为dp[i]加上dp[i-num]和dp[i+num]的数值
# https://leetcode.cn/problems/target-sum
# 这题也可以去看一下灵神的记忆化搜索的题解
class Solution(object):
    def findTargetSumWays(self, nums, target):
        total_sum = sum(nums)
        if abs(target) > total_sum:
            return 0  # target 超出可达范围

        offset = total_sum  # 偏移量，使负数索引转换为非负
        dp = [0] * (2 * total_sum + 1)  # dp数组大小为 [-total_sum, total_sum] 的范围
        dp[offset] = 1  # 初始状态，和为 0 的方法数为 1

        for num in nums:
            next_dp = [0] * (2 * total_sum + 1)  # 复制 dp，避免状态覆盖
            for i in range(2 * total_sum + 1):
                if dp[i] > 0:  # 只有 dp[i] > 0 才有意义
                    if 0 <= i + num < 2 * total_sum + 1:
                        next_dp[i + num] += dp[i]  # 加 num
                    if 0 <= i - num < 2 * total_sum + 1:
                        next_dp[i - num] += dp[i]  # 减 num
            dp = next_dp  # 更新 dp

        return dp[target + offset]  # 目标值需要加上偏移量才能索引