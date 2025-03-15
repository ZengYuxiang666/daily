# 看到什么样的题目用单调栈？ 求一个数组中的元素单调递减时突然出现比他大的元素时的距离
# 求一个数组中相聚最长的“阶梯”
# https://leetcode.cn/problems/maximum-width-ramp
# 因为先求出来了递减的数字下标，最后的一个肯定是数组中最小的一个，再把数组反方向遍历得最长

# 求最长
# 表现良好的最长时间段和这题相似

class Solution(object):
    def maxWidthRamp(self, nums):
        # Step 1: 构建单调递减栈，存储索引
        stack = []
        for i in range(len(nums)):
            if not stack or nums[i] < nums[stack[-1]]:
                stack.append(i)
        max_width = 0
        # Step 2: 从右到左遍历，更新最大宽度
        for j in range(len(nums) - 1, -1, -1):
            while stack and nums[stack[-1]] <= nums[j]:
                i = stack.pop()
                max_width = max(max_width, j - i)

        return max_width