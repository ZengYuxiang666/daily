# 维护一个单调递增的单调栈就是最小数字
# https://leetcode.cn/problems/remove-k-digits
class Solution:
    def removeKdigits(self, num: str, k: int) -> str:
        stack = []

        for digit in num:
            # 如果当前数字比栈顶元素小，并且还有删除次数 k，则移除栈顶
            while k > 0 and stack and stack[-1] > digit:
                stack.pop()
                k -= 1
            stack.append(digit)

        # 如果还需要删除元素，继续从栈顶移除
        while k > 0:
            stack.pop()
            k -= 1

        # 去掉前导零并返回结果
        result = ''.join(stack).lstrip('0')

        # 如果结果为空，返回 "0"
        return result if result else "0"