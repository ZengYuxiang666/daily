# 求最长，最大，用for循环移动right指针，当不满足条件时用while循环移动left指针，直到满足条件为止，每次循环更新最大长度
# https://leetcode.cn/problems/length-of-longest-subarray-with-at-most-k-frequency
class Solution(object):
    def maxSubarrayLength(self, nums, k):
        dic = {}
        left = 0
        max_len = 0

        for right in range(len(nums)):
            if nums[right] not in dic:
                dic[nums[right]] = 0
            dic[nums[right]] += 1

            while dic[nums[right]] > k:
                dic[nums[left]] -= 1
                left += 1

            max_len = max(max_len, right - left + 1)

        return max_len
