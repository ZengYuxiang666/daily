# 当一个问题要重复计算同一个子问题，就可以用动态规划，也就是记忆化搜索
# 这个题目返回的是一个最长子序列的数组，要记住这个思路
# https://leetcode.cn/problems/largest-divisible-subset
from functools import cache
from typing import List
class Solution:
    def largestDivisibleSubset(self, nums: List[int]) -> List[int]:
        nums.sort()  # 排序以简化判断
        n = len(nums)
        @cache
        def dfs(i):
            """返回以 nums[i] 为起点的最大整除子集"""
            max_subset = [nums[i]]

            for j in range(i):
                if nums[i] % nums[j] == 0:
                    subset = dfs(j)
                    if len(subset) + 1 > len(max_subset):
                        max_subset = subset + [nums[i]]
            return max_subset
        # 找出最大整除子集
        largest_subset = []

        for i in range(n):
            subset = dfs(i)
            if len(subset) > len(largest_subset):
                largest_subset = subset

        return largest_subset