# 双指针，right往右走，当不满足条件时left往右走一格，total+=right-left+1
# https://leetcode.cn/problems/subarray-product-less-than-k
class Solution(object):
    def numSubarrayProductLessThanK(self, nums, k):
        if k <= 1:
            return 0

        count = 0
        product = 1
        left = 0

        for right in range(len(nums)):
            product *= nums[right]

            while product >= k:
                product //= nums[left]
                left += 1

            count += (right - left + 1)

        return count