# https://leetcode.cn/problems/longest-nice-subarray
# 一个把与和或，异或的性质都用起来的题，先用or_ & x判断是否有重复1，假如有重复1，用or_ ^= nums[left]移除nums[left]的1，移动左指针
# 假如为0，用or_ |= x将right的1加入or_中
class Solution:
    def longestNiceSubarray(self, nums: List[int]) -> int:
        ans = left = or_ = 0
        for right, x in enumerate(nums):
            while or_ & x:  # 有交集
                or_ ^= nums[left]  # 从 or_ 中去掉集合 nums[left]
                left += 1
            or_ |= x  # 把集合 x 并入 or_ 中
            ans = max(ans, right - left + 1)
        return ans