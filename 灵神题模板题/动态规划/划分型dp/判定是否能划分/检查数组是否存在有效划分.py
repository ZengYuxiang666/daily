# 如果存在就返回True，如果不存在就返回False，
# https://leetcode.cn/problems/check-if-there-is-a-valid-partition-for-the-array
from functools import cache
from typing import List
class Solution:
    def validPartition(self, nums: List[int]) -> bool:
        n = len(nums)
        memo = [-1] * n
        # 从后往前进行 DFS
        @cache
        def dfs(i):
            if i < 0:  # 已经遍历完所有元素，表示可以成功划分
                return True
            # 长度为2的子数组
            if i >= 1 and nums[i] == nums[i - 1]:
                if dfs(i - 2):
                    return True
            # 长度为3的子数组
            if i >= 2:
                # 三个相等
                if nums[i] == nums[i - 1] == nums[i - 2]:
                    if dfs(i - 3):
                        return True

                # 三个连续递增
                if nums[i] - 1 == nums[i - 1] and nums[i - 1] - 1 == nums[i - 2]:
                    if dfs(i - 3):
                        return True
            return False

        return dfs(n - 1)
p = Solution()
print(p.validPartition([1,1,1,2]))