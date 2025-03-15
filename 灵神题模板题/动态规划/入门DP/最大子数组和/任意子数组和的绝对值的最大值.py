# 求一个子数组的绝对值的最大值可以维护两个dp数组，一个为最大dp数组一个为最小dp数组
# https://leetcode.cn/problems/maximum-absolute-sum-of-any-subarray
class Solution(object):
    def maxAbsoluteSum(self, nums):
        n = len(nums)
        dp_max = [0] * n  # 以 i 结尾的最大子数组和
        dp_min = [0] * n  # 以 i 结尾的最小子数组和

        dp_max[0] = dp_min[0] = nums[0]
        max_sum = min_sum = nums[0]

        for i in range(1, n):
            dp_max[i] = max(nums[i], dp_max[i - 1] + nums[i])
            dp_min[i] = min(nums[i], dp_min[i - 1] + nums[i])

            max_sum = max(max_sum, dp_max[i])
            min_sum = min(min_sum, dp_min[i])

        return max(abs(max_sum), abs(min_sum))