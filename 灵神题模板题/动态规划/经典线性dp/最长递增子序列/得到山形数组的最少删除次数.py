# 定义两个函数，一个是求最长递增子序列，一个是求最长递减子序列，然后枚举除了两边的每个i
# https://leetcode.cn/problems/minimum-number-of-removals-to-make-mountain-array
from functools import cache
from typing import List
class Solution:
    def minimumMountainRemovals(self, nums: List[int]) -> int:
        @cache
        def dfs_plus(i):  # 求最长递增子序列
            res = 0
            for j in range(i):
                if nums[j] < nums[i]:
                    res = max(res, dfs_plus(j))
            return res + 1
        @cache
        def dfs_down(i):  # 求最长递减子序列
            res = 0
            for j in range(i + 1, len(nums)):  # 从i之后的元素开始找
                if nums[j] < nums[i]:
                    res = max(res, dfs_down(j))
            return res + 1
        result = 0
        for i in range(1, len(nums) - 1):  # 山峰不能在两端
            if dfs_plus(i) > 1 and dfs_down(i) > 1:
                result = max(result, dfs_plus(i) + dfs_down(i) - 1)
        return len(nums) - result