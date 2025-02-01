# 一个数与另一个数异或 a^b = num，a = mun^b
# https://leetcode.cn/problems/maximum-xor-for-each-query
class Solution(object):
    def getMaximumXor(self, nums, maximumBit):
        # 排序的 nums 是有序的，我们需要最大化的 xor_sum
        nums_xor = 0
        for num in nums:
            nums_xor ^= num  # 计算所有元素的初始 XOR 值

        max_value = (1 << maximumBit) - 1  # 最大化的 XOR mask
        answer = []

        # 逆序删除元素，每次更新 nums_xor 并计算 k
        for num in reversed(nums):
            answer.append(max_value ^ nums_xor)
            nums_xor ^= num  # 移除最后一个元素的影响

        return answer
