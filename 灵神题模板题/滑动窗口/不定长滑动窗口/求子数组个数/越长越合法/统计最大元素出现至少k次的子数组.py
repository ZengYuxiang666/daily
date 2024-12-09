# 用for循环不断移动right指针，满足条件时total+=left，然后left指针左移，total不能+=len(nums)-right,因为移动的是右指针，这么加会重复计数
# 因为是越长越合法，当合法时该数组往左伸的数组肯定成立，所以total+=left
# https://leetcode.cn/problems/count-subarrays-where-max-element-appears-at-least-k-times
class Solution(object):
    def countSubarrays(self, nums, k):
        total = 0
        left = 0
        max_num = max(nums)
        result = 0
        for right in range(len(nums)):
            if nums[right] == max_num:
                total += 1
            while total == k:
                if nums[left] == max_num:
                    total -= 1
                left += 1
            result += left
        return result

