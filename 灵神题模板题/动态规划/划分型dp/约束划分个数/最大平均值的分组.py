# 假如约束了个数，只要在dfs中加个k参数就是
# https://leetcode.cn/problems/largest-sum-of-averages
from functools import cache
from typing import List

class Solution:
    def largestSumOfAverages(self, nums: List[int], k: int) -> float:
        n = len(nums)
        # 计算前缀和
        prefix_sum = [0] * (n + 1)
        for i in range(1, n + 1):
            prefix_sum[i] = prefix_sum[i - 1] + nums[i - 1]

        # 计算出nums[i:j+1]的平均值
        @cache
        def average(i, j):
            if j < i:
                return 0
            return (prefix_sum[j+1] - prefix_sum[i]) / (j - i + 1)

        @cache
        def dfs(i, k):
            # Base case: 当k == 1时，返回[0:i+1]的平均值
            if k == 1:
                return average(0, i)

            # 递归计算最大分数
            max_avg = float('-inf')
            for j in range(i+1):
                max_avg = max(max_avg, dfs(j-1, k - 1) + average(j, i))

            return max_avg

        return dfs(n-1, k)

# 示例测试
p = Solution()
print(p.largestSumOfAverages(nums = [1,2,3,4,5,6,7], k = 4))  # 输出应该接近 20.0