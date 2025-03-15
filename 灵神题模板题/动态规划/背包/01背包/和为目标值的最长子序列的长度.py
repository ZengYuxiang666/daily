# 当是递推关系，且nums中的数据每个只能用一次时要用01背包，感觉关键还是在状态转移方程
# 有一个数组然后有一个target值要想到是背包问题
# https://leetcode.cn/problems/length-of-the-longest-subsequence-that-sums-to-target
from typing import List
class Solution:
    def lengthOfLongestSubsequence(self, nums: List[int], target: int) -> int:
        dp = [-1] * (target + 1)  # -1 表示无法凑成该和
        dp[0] = 0  # 凑成 0 需要 0 个数
        for num in nums:
            for j in range(target, num - 1, -1):  # 逆序遍历
                if dp[j - num] != -1:  # 确保 j-num 可达
                    dp[j] = max(dp[j], dp[j - num] + 1)

        return dp[target]