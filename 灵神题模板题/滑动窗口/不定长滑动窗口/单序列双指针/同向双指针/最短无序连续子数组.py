# 同向双指针就和之前做的那样
# https://leetcode.cn/problems/shortest-unsorted-continuous-subarray
class Solution(object):
    def findUnsortedSubarray(self, nums):
        n = len(nums)
        left = 0
        right = n - 1

        # 找到第一个不是递增的地方
        while left < n - 1 and nums[left] <= nums[left + 1]:
            left += 1

        # 如果整个数组有序，直接返回 0
        if left == n - 1:
            return 0

        # 找到从右往左第一个不是递减的地方
        while right > 0 and nums[right] >= nums[right - 1]:
            right -= 1

        # 找到无序子数组中的最大值和最小值
        sub_min = min(nums[left:right + 1])
        sub_max = max(nums[left:right + 1])

        # 扩展左边界，使得无序子数组的最小值可以放到正确的位置
        while left > 0 and nums[left - 1] > sub_min:
            left -= 1

        # 扩展右边界，使得无序子数组的最大值可以放到正确的位置
        while right < n - 1 and nums[right + 1] < sub_max:
            right += 1

        # 返回无序子数组的长度
        return right - left + 1