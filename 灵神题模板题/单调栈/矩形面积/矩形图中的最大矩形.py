# 这个题的核心是单调栈，假如一个不断上升矩形突然出现一个小的，就更新他的最大面积
# https://leetcode.cn/problems/largest-rectangle-in-histogram
class Solution(object):
    def largestRectangleArea(self, heights):
        stack = []
        max_area = 0
        heights.append(0)  # 在末尾添加一个 0，确保栈中所有元素都能被处理

        for i in range(len(heights)):
            while stack and heights[i] < heights[stack[-1]]:
                h = heights[stack.pop()]
                w = i if not stack else i - stack[-1] - 1
                max_area = max(max_area, h * w)
            stack.append(i)

        return max_area
"""
初始化一个空栈 stack。
遍历每个柱子（索引从 0 到 n-1）：
如果栈为空或当前柱子的高度大于栈顶柱子的高度，直接将当前柱子索引压入栈中。
否则，弹出栈顶元素，计算面积，更新最大面积。
最后处理栈中剩余的柱子。
"""