# 核心是维护一个递增的单调栈，但把弹出条件设置成了要大于等于k个才能弹出
# https://leetcode.cn/problems/find-the-most-competitive-subsequence
class Solution(object):
    def mostCompetitive(self, nums, k):
        stack = []
        for i in range(len(nums)):
            # 确保只有在还能满足最终长度为 `k` 时才弹出元素
            while stack and nums[i] < stack[-1] and len(stack) + len(nums) - i - 1 >= k:
                stack.pop()
            stack.append(nums[i])

        # 返回前 `k` 个元素
        return stack[:k]
