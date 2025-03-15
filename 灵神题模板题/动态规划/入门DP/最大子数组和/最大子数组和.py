# 没有什么说的，看代码就是了，关键在状态转移方程
# https://leetcode.cn/problems/maximum-subarray
from typing import List


class Solution:
    def maxSubArray(self, nums: List[int]) -> int:
        n = len(nums)
        dp = [0] * n  # dp[i] 代表以 i 结尾的最大子数组和
        dp[0] = nums[0]  # 初始状态

        max_sum = dp[0]  # 记录全局最大子数组和

        for i in range(1, n):
            dp[i] = max(nums[i], dp[i - 1] + nums[i])  # 状态转移方程
            max_sum = max(max_sum, dp[i])  # 更新最大值

        return max_sum