# 将一个数组分成k份，没份求一个最长递增子序列，每份数组的长度减去最长的长度等于这一份的最小操作次数，最后每份相加
# https://leetcode.cn/problems/minimum-operations-to-make-the-array-k-increasing
from functools import cache
from typing import List
class Solution:
    def kIncreasing(self, arr: List[int], k: int) -> int:
        @cache
        def dfs(i: int) -> int:
            """返回以 i 结尾的最长非递减子序列长度"""
            res = 0  # 至少当前元素本身可以形成长度为1的子序列
            for j in range(i - k, -1, -k):  # 向前跳跃k步
                if arr[j] <= arr[i]:  # 满足非递减条件
                    res = max(res, dfs(j))
            return res + 1
        n = len(arr)
        result = 0
        # 遍历每个子序列
        for i in range(k):
            max_len = 0
            for j in range(i, n, k):  # 按步长跳跃遍历子序列
                max_len = max(max_len, dfs(j))
            # 子序列长度 - 最长非递减子序列长度 = 修改次数
            result += (len(range(i, n, k)) - max_len)
        return result