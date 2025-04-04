# 就是选或者不选的问题，如果必须选一个，可以将当前选择的两个也加入状态转移方程
# 最长公共子序列题目，即从后往前是有顺序的
# https://leetcode.cn/problems/max-dot-product-of-two-subsequences
from functools import cache
from typing import List
class Solution:
    def maxDotProduct(self, nums1: List[int], nums2: List[int]) -> int:
        @cache
        def dfs(i, j):
            if i < 0 or j < 0:
                return float('-inf')
            return max(nums1[i] * nums2[j] + dfs(i - 1, j - 1), dfs(i - 1, j), dfs(i, j - 1), nums1[i] * nums2[j])

        return dfs(len(nums1) - 1, len(nums2) - 1)