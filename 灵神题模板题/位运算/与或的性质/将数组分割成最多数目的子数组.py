# AND运算，只会减小数值或不变
# https://leetcode.cn/problems/split-array-into-maximum-number-of-subarrays
# 数组的最小数值肯定为整个数组的and值，当最小值不为0时肯定只有一个子数组，因为假如最小值不为0，不可能有两个子数组值相等，因为相等的话
# 整个数组的and值应该为0才对
class Solution:
    def maxSubarrays(self, nums) -> int:
        ans = 0
        a = -1  # -1 就是 111...1，和任何数 AND 都等于那个数
        for x in nums:
            a &= x
            if a == 0:
                ans += 1  # 分割
                a = -1
        return max(ans, 1)  # 如果 ans=0 说明所有数的 and>0，答案为 1
